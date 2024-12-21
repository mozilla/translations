/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * This file is a pared-down version of `translations-engine.sys.mjs` from the Firefox source code.
 * https://searchfox.org/mozilla-central/rev/450aacd753c98b3200f120ed4340e1ed53b7ff47/toolkit/components/translations/content/translations-engine.sys.mjs
 *
 * This version excludes the Firefox-specific complexity and mechanisms that are required for integration into
 * the Firefox Translations ecosystem. This allows us to test the WASM bindings directly within development
 * environment in which they are generated, before they are vendored into Firefox.
 *
 * The worker used within this file runs in a Node.js worker_threads environment, but is designed to simulate
 * the same code paths of communicating with Web Worker in a browser environment. A subset of the Web Worker
 * functionality is simulated to provide the required API surface.
 *
 * @see {WebWorkerSimulator}
 */

/**
 * @typedef {import('./../../bindings/bergamot-translator.d.ts').TranslationModelPayload} TranslationModelPayload
 */

import path from "path";
import fs from "fs/promises";
import { Worker } from "node:worker_threads";

const MODELS_PATH = "./models";
const WORKER_PATH = "./engine/translations-engine.worker.mjs";
const WASM_PATH = "./generated/bergamot-translator.wasm";
const PIVOT = "en";

export class TranslationsEngine {
  #worker;
  #isReady;
  #isReadyResolve;
  #isReadyReject;
  #currentTranslationResolve = null;
  #currentTranslationReject = null;

  /**
   * Constructs a new Translator instance.
   *
   * @param {string} sourceLanguage - The source language code (e.g., 'es').
   * @param {string} targetLanguage - The target language code (e.g., 'fr').
   */
  constructor(sourceLanguage, targetLanguage) {
    this.#worker = new Worker(WORKER_PATH);

    this.#worker.on("message", (data) => this.#handleMessage({ data }));
    this.#worker.on("error", (error) => this.#handleError({ error }));

    this.#isReady = this.#initWorker(sourceLanguage, targetLanguage);
  }

  /**
   * Private method to initialize the worker by reading necessary files and sending the initialization message.
   *
   * @returns {Promise<void>}
   */
  async #initWorker(sourceLanguage, targetLanguage) {
    try {
      const wasmBuffer = await fs.readFile(WASM_PATH);
      const translationModelPayloads =
        await this.#prepareTranslationModelPayloads(
          sourceLanguage,
          targetLanguage,
        );

      // Return a promise that resolves or rejects based on worker messages
      const isReadyPromise = new Promise((resolve, reject) => {
        this.#isReadyResolve = resolve;
        this.#isReadyReject = reject;
      });

      this.#worker.postMessage({
        type: "initialize",
        enginePayload: {
          bergamotWasmArrayBuffer: wasmBuffer.buffer,
          translationModelPayloads,
        },
      });

      return isReadyPromise;
    } catch (error) {
      throw new Error(`
        🚨 Failed to read one or more files required for translation 🚨

        ${error}

        ⏩ NEXT STEPS ⏩

        To ensure that test dependencies are properly setup, please run the following command:

        ❯ task inference-test-wasm
      `);
    }
  }

  /**
   * Private helper method to prepare the language model files.
   *
   * @param {string} sourceLanguage - The source language code.
   * @param {string} targetLanguage - The target language code.
   * @returns {Promise<Array<TranslationModelPayload>>} - An array of translation model payloads.
   */
  async #prepareTranslationModelPayloads(sourceLanguage, targetLanguage) {
    let translationModelPayloadPromises;

    if (sourceLanguage === PIVOT || targetLanguage === PIVOT) {
      translationModelPayloadPromises = [
        this.#loadTranslationModelPayload(sourceLanguage, targetLanguage),
      ];
    } else {
      translationModelPayloadPromises = [
        this.#loadTranslationModelPayload(sourceLanguage, PIVOT),
        this.#loadTranslationModelPayload(PIVOT, targetLanguage),
      ];
    }

    return Promise.all(translationModelPayloadPromises);
  }

  /**
   * Private helper method to load language model files.
   *
   * @param {string} sourceLanguage - The source language code.
   * @param {string} targetLanguage - The target language code.
   * @returns {Promise<TranslationModelPayload>} - An object containing the data required to construct a translation model.
   */
  async #loadTranslationModelPayload(sourceLanguage, targetLanguage) {
    const langPairDirectory = `${MODELS_PATH}/${sourceLanguage}${targetLanguage}`;

    const lexPath = path.join(
      langPairDirectory,
      `lex.50.50.${sourceLanguage}${targetLanguage}.s2t.bin`,
    );
    const modelPath = path.join(
      langPairDirectory,
      `model.${sourceLanguage}${targetLanguage}.intgemm.alphas.bin`,
    );
    const vocabPath = path.join(
      langPairDirectory,
      `vocab.${sourceLanguage}${targetLanguage}.spm`,
    );

    const [lexBuffer, modelBuffer, vocabBuffer] = await Promise.all([
      fs.readFile(lexPath),
      fs.readFile(modelPath),
      fs.readFile(vocabPath),
    ]);

    return {
      sourceLanguage,
      targetLanguage,
      languageModelFiles: {
        model: { buffer: modelBuffer },
        lex: { buffer: lexBuffer },
        vocab: { buffer: vocabBuffer },
      },
    };
  }

  /**
   * Private method to handle incoming messages from the worker.
   *
   * @param {MessageEvent} event - The message event from the worker.
   */
  #handleMessage(event) {
    const { data } = event;

    switch (data.type) {
      case "initialization-success": {
        this.#isReadyResolve && this.#isReadyResolve();
        break;
      }
      case "initialization-error": {
        this.#isReadyReject && this.#isReadyReject(new Error(data.error));
        break;
      }
      case "translation-response": {
        if (this.#currentTranslationResolve) {
          this.#currentTranslationResolve(data.targetText);
          this.#clearCurrentTranslation();
        }
        break;
      }
      case "translation-error": {
        if (this.#currentTranslationReject) {
          this.#currentTranslationReject(new Error(data.error.message));
          this.#clearCurrentTranslation();
        }
        break;
      }
      default: {
        console.warn(`Unknown message type: ${data.type}`);
      }
    }
  }

  /**
   * Private method to handle errors from the worker.
   *
   * @param {ErrorEvent} error - The error event from the worker.
   */
  #handleError(error) {
    if (this.#isReadyReject) {
      this.#isReadyReject(error);
    }
    if (this.#currentTranslationReject) {
      this.#currentTranslationReject(error);
      this.#clearCurrentTranslation();
    }
  }

  /**
   * Translates the given source text.
   *
   * @param {string} sourceText - The text to translate.
   * @param {boolean} [isHTML=false] - Indicates if the source text is HTML.
   * @returns {Promise<string>} - The translated text.
   */
  async translate(sourceText, isHTML = false) {
    await this.#isReady;

    return new Promise((resolve, reject) => {
      this.#currentTranslationResolve = resolve;
      this.#currentTranslationReject = reject;

      // Send translation request
      this.#worker.postMessage({
        type: "translation-request",
        sourceText,
        isHTML,
      });
    });
  }

  /**
   * Clears the current translation promise handlers.
   */
  #clearCurrentTranslation() {
    this.#currentTranslationResolve = null;
    this.#currentTranslationReject = null;
  }

  /**
   * Terminates the worker and cleans up resources.
   */
  terminate() {
    if (this.#worker) {
      this.#clearCurrentTranslation();
      this.#worker.terminate();
      this.#worker.onmessage = null;
      this.#worker.onerror = null;
      this.#worker = null;
    }
  }
}
