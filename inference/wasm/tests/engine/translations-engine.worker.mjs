/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * This file is a pared-down version of `translations-engine.worker.js` within the Firefox source code:
 * https://searchfox.org/mozilla-central/rev/450aacd753c98b3200f120ed4340e1ed53b7ff47/toolkit/components/translations/content/translations-engine.worker.js
 *
 * This version excludes the Firefox-specific complexity and mechanisms that are required for integration into
 * the Firefox Translations ecosystem. This allows us to test the WASM bindings directly within development
 * environment in which they are generated, before they are vendored into Firefox.
 *
 * This file runs within a Node.js worker_threads environment, but is designed to simulate the same code paths
 * of loading and running our generated code in a Web Workers environment. A subset of the WorkerGlobalScope
 * functionality is simulated to provide the required API surface.
 *
 * @see {WorkerGlobalScopeSimulator}
 */

/**
 * Importing types from the TypeScript declaration file using JSDoc.
 * This allows us to use the types in our JavaScript code for better documentation and tooling support.
 *
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.BergamotModule} BergamotModule
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.TranslationModel} TranslationModel
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.BlockingService} BlockingService
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.VectorString} VectorString
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.VectorResponseOptions} VectorResponseOptions
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.VectorResponse} VectorResponse
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.ResponseOptions} ResponseOptions
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.Response} Response
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.AlignedMemory} AlignedMemory
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').Bergamot.AlignedMemoryList} AlignedMemoryList
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').TranslationModelPayload} TranslationModelPayload
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').LanguageTranslationModelFilesAligned} LanguageTranslationModelFilesAligned
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').TranslationsEnginePayload} TranslationsEnginePayload
 */

import WorkerGlobalScopeSimulator from "./worker-global-scope-simulator.mjs";

/**
 * Simulate the WorkerGlobalScope from a browser environment within Node.js
 * https://developer.mozilla.org/en-US/docs/Web/API/WorkerGlobalScope
 */
const self = new WorkerGlobalScopeSimulator();

try {
  /* global loadBergamot */
  self.importScripts("./generated/bergamot-translator.js");
} catch (error) {
  self.postMessage({ type: "initialization-error", error: error.message });
}

/**
 * Constants defining alignment requirements for different model files.
 *
 * @type {Record<string, number>}
 */
const MODEL_FILE_ALIGNMENTS = {
  model: 256,
  lex: 64,
  vocab: 64,
  qualityModel: 64,
  srcvocab: 64,
  trgvocab: 64,
};

/**
 * Event listener for handling initialization messages.
 */
self.addEventListener("message", handleInitializationMessage);

/**
 * Handles the initialization message to set up the translation engine.
 *
 * @param {MessageEvent} event - The message event containing initialization data.
 */
async function handleInitializationMessage(event) {
  const { data } = event;
  if (data.type !== "initialize") {
    return;
  }

  try {
    /** @type {TranslationsEnginePayload} */
    const enginePayload = data.enginePayload;
    const { bergamotWasmArrayBuffer, translationModelPayloads } = enginePayload;

    /** @type {BergamotModule} */
    const bergamot = await BergamotUtils.initializeWasm(
      bergamotWasmArrayBuffer,
    );

    const engine = new Engine(bergamot, translationModelPayloads);

    // Handle translation requests
    self.addEventListener("message", async (messageEvent) => {
      const messageData = messageEvent.data;

      if (messageData.type === "translation-request") {
        const { sourceText, isHTML } = messageData;

        try {
          const { whitespaceBefore, whitespaceAfter, cleanedSourceText } =
            cleanText(sourceText);

          const targetText =
            whitespaceBefore +
            engine.translate(cleanedSourceText, isHTML) +
            whitespaceAfter;

          self.postMessage({
            type: "translation-response",
            targetText,
          });
        } catch (error) {
          self.postMessage({
            type: "translation-error",
            error: { message: error.message, stack: error.stack },
          });
        }
      }
    });

    self.postMessage({ type: "initialization-success" });
  } catch (error) {
    self.postMessage({
      type: "initialization-error",
      error: error.message,
    });
  }

  self.removeEventListener("message", handleInitializationMessage);
}

/**
 * The Engine class handles translation using the Bergamot WASM module.
 */
class Engine {
  /**
   * The initialized Bergamot WASM module.
   *
   * @type {BergamotModule}
   */
  #bergamot;

  /**
   * An array of translation models.
   *
   * @type {TranslationModel[]}
   */
  #languageTranslationModels;

  /**
   * The translation service used to perform translations.
   *
   * @type {BlockingService}
   */
  #translationService;

  /**
   * Constructs the Engine instance.
   *
   * @param {BergamotModule} bergamot - Initialized Bergamot module.
   * @param {TranslationModelPayload[]} translationModelPayloads - Payloads to construct translation models.
   */
  constructor(bergamot, translationModelPayloads) {
    this.#bergamot = bergamot;

    this.#languageTranslationModels = translationModelPayloads.map(
      (modelFiles) => {
        return BergamotUtils.constructSingleTranslationModel(
          bergamot,
          modelFiles,
        );
      },
    );

    this.#translationService = new bergamot.BlockingService({ cacheSize: 0 });
  }

  /**
   * Translates the given source text.
   *
   * @param {string} sourceText - Text to translate.
   * @param {boolean} isHTML - Indicates if the text contains HTML.
   * @returns {string} Translated text.
   */
  translate(sourceText, isHTML) {
    return this.#syncTranslate(sourceText, isHTML);
  }

  /**
   * Performs synchronous translation.
   *
   * @param {string} sourceText - Text to translate.
   * @param {boolean} isHTML - Indicates if the text contains HTML.
   * @returns {string} Translated text.
   */
  #syncTranslate(sourceText, isHTML) {
    /** @type {VectorResponse} */
    let responses;
    const { messages, options } = BergamotUtils.getTranslationArgs(
      this.#bergamot,
      sourceText,
      isHTML,
    );

    try {
      if (messages.size() === 0) {
        return "";
      }

      if (this.#languageTranslationModels.length === 1) {
        responses = this.#translationService.translate(
          this.#languageTranslationModels[0],
          messages,
          options,
        );
      } else if (this.#languageTranslationModels.length === 2) {
        responses = this.#translationService.translateViaPivoting(
          this.#languageTranslationModels[0],
          this.#languageTranslationModels[1],
          messages,
          options,
        );
      } else {
        throw new Error(
          "Too many models were provided to the translation worker.",
        );
      }

      const targetText = responses.get(0).getTranslatedText();
      return targetText;
    } finally {
      messages.delete();
      options.delete();
      responses?.delete();
    }
  }
}

/**
 * Utility class for Bergamot WASM operations.
 */
class BergamotUtils {
  /**
   * Constructs a single translation model.
   *
   * @param {BergamotModule} bergamot - Initialized Bergamot module.
   * @param {TranslationModelPayload} translationModelPayload - The payload to construct a translation model.
   * @returns {TranslationModel} Constructed translation model.
   */
  static constructSingleTranslationModel(bergamot, translationModelPayload) {
    const { sourceLanguage, targetLanguage, languageModelFiles } =
      translationModelPayload;

    const { model, lex, vocab, qualityModel, srcvocab, trgvocab } =
      BergamotUtils.allocateModelMemory(bergamot, languageModelFiles);

    /** @type {AlignedMemoryList} */
    const vocabList = new bergamot.AlignedMemoryList();

    if (vocab) {
      vocabList.push_back(vocab);
    } else if (srcvocab && trgvocab) {
      vocabList.push_back(srcvocab);
      vocabList.push_back(trgvocab);
    } else {
      throw new Error("Vocabulary key is not found.");
    }

    const config = BergamotUtils.generateTextConfig({
      "beam-size": "1",
      normalize: "1.0",
      "word-penalty": "0",
      "max-length-break": "128",
      "mini-batch-words": "1024",
      workspace: "128",
      "max-length-factor": "2.0",
      "skip-cost": "true",
      "cpu-threads": "0",
      quiet: "true",
      "quiet-translation": "true",
      "gemm-precision": "int8shiftAlphaAll",
      alignment: "soft",
    });

    return new bergamot.TranslationModel(
      sourceLanguage,
      targetLanguage,
      config,
      model,
      lex,
      vocabList,
      qualityModel ?? null,
    );
  }

  /**
   * Allocates aligned memory for the model files.
   *
   * @param {BergamotModule} bergamot - Initialized Bergamot module.
   * @param {LanguageTranslationModelFiles} languageModelFiles - Model files for translation.
   * @returns {LanguageTranslationModelFilesAligned} Allocated memory for each file type.
   */
  static allocateModelMemory(bergamot, languageModelFiles) {
    /** @type {LanguageTranslationModelFilesAligned} */
    const results = {};

    for (const [fileType, file] of Object.entries(languageModelFiles)) {
      const alignment = MODEL_FILE_ALIGNMENTS[fileType];
      if (!alignment) {
        throw new Error(`Unknown file type: "${fileType}"`);
      }

      /** @type {AlignedMemory} */
      const alignedMemory = new bergamot.AlignedMemory(
        file.buffer.byteLength,
        alignment,
      );
      alignedMemory.getByteArrayView().set(new Uint8Array(file.buffer));

      results[fileType] = alignedMemory;
    }

    return results;
  }

  /**
   * Initializes the Bergamot WASM module.
   *
   * @param {ArrayBuffer} wasmBinary - The WASM binary data.
   * @returns {Promise<BergamotModule>} Resolves with the initialized Bergamot module.
   */
  static initializeWasm(wasmBinary) {
    return new Promise((resolve, reject) => {
      /** @type {BergamotModule} */
      const bergamot = loadBergamot({
        INITIAL_MEMORY: 234_291_200,
        print: () => {},
        // Uncomment this line to display logs in tests.
        // To show logs, run with the --runner=basic flag.
        // print: (...args) => console.log(...args),
        onAbort() {
          reject(new Error("Error loading Bergamot WASM module."));
        },
        onRuntimeInitialized: () => {
          try {
            resolve(bergamot);
          } catch (e) {
            reject(e);
          }
        },
        wasmBinary,
      });
    });
  }

  /**
   * Generates a configuration string for the Marian translation service.
   *
   * @param {Record<string, string>} config - Configuration key-value pairs.
   * @returns {string} Formatted configuration string.
   */
  static generateTextConfig(config) {
    const indent = "            ";
    let result = "\n";

    for (const [key, value] of Object.entries(config)) {
      result += `${indent}${key}: ${value}\n`;
    }

    return result + indent;
  }

  /**
   * Prepares translation arguments for the Bergamot module.
   *
   * @param {BergamotModule} bergamot - Initialized Bergamot module.
   * @param {string} sourceText - Text to translate.
   * @param {boolean} isHTML - Indicates if the text contains HTML.
   * @returns {{messages: VectorString, options: VectorResponseOptions}} Prepared messages and options.
   */
  static getTranslationArgs(bergamot, sourceText, isHTML) {
    /** @type {VectorString} */
    const messages = new bergamot.VectorString();
    /** @type {VectorResponseOptions} */
    const options = new bergamot.VectorResponseOptions();

    if (sourceText) {
      messages.push_back(sourceText);
      options.push_back({
        qualityScores: false,
        alignment: true,
        html: isHTML,
      });
    }

    return { messages, options };
  }
}

/**
 * Regular expression to match whitespace before and after the text.
 *
 * @type {RegExp}
 */
const whitespaceRegex = /^(\s*)(.*?)(\s*)$/s;

/**
 * Cleans the text before translation by preserving surrounding whitespace and removing soft hyphens.
 *
 * @param {string} sourceText - The original text to clean.
 * @returns {{whitespaceBefore: string, whitespaceAfter: string, cleanedSourceText: string}} Contains whitespace before, after, and the cleaned text.
 */
function cleanText(sourceText) {
  const result = whitespaceRegex.exec(sourceText);
  if (!result) {
    throw new Error("Failed to match whitespace in the source text.");
  }
  const whitespaceBefore = result[1];
  const whitespaceAfter = result[3];
  let cleanedSourceText = result[2];

  // Remove all soft hyphens from the text.
  cleanedSourceText = cleanedSourceText.replaceAll(/\u00AD/g, "");

  // At the time of writing, the Intl.Segmenter has a less-than-ideal segmentation pattern when
  // a Left Double Quotation Mark (U+201C) is preceded by a full-width punctuation mark, in which
  // it fails to segment the quotation mark with the sentence it logically belongs to.
  //
  // Example Source Text:
  //   - 这是第一句话。“这是第二句话。”
  //
  // Expected Segmentation:
  //   - Object { index: 0, segment: 这是第一句话。 }
  //   - Object { index: 7, segment: “这是第二句话。” }
  //
  // Actual Segmentation:
  //   - Object { index: 0, segment: 这是第一句话。“ }
  //   - Object { index: 8, segment: 这是第二句话。” }
  //
  // By inserting a space between the full-width punctuation and the Left Double Quotation Mark,
  // we can trick the segmenter into breaking the sentence at the correct location.
  //
  // This code may be able to be removed with further upstream improvements to Intl.Segmenter.
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/Segmenter
  cleanedSourceText = cleanedSourceText.replaceAll(/([。！？])“/g, "$1 “");

  return { whitespaceBefore, whitespaceAfter, cleanedSourceText };
}
