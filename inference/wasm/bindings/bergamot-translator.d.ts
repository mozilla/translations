
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

export namespace Bergamot {
  /**
   * The main module that is returned from bergamot-translator.js.
   */
  export interface BergamotModule {
    BlockingService: typeof BlockingService;
    AlignedMemoryList: typeof AlignedMemoryList;
    TranslationModel: typeof TranslationModel;
    AlignedMemory: typeof AlignedMemory;
    VectorResponseOptions: typeof VectorResponseOptions;
    VectorString: typeof VectorString;
  }

  /**
   * This class represents a C++ std::vector. The implementations will extend from it.
   */
  export class Vector<T> {
    size(): number;
    get(index: number): T;
    push_back(item: T);
  }

  export class VectorResponse extends Vector<Response> {}
  export class VectorString extends Vector<string> {}
  export class VectorResponseOptions extends Vector<ResponseOptions> {}
  export class AlignedMemoryList extends Vector<AlignedMemory> {}

  /**
   * A blocking (e.g. non-threaded) translation service, via Bergamot.
   */
  export class BlockingService {
    /**
     * Translate multiple messages in a single synchronous API call using a single model.
     */
    translate(
      translationModel,
      vectorSourceText: VectorString,
      vectorResponseOptions: VectorResponseOptions
    ): VectorResponse;

    /**
     * Translate by pivoting between two models
     *
     * For example to translate "fr" to "es", pivot using "en":
     *   "fr" to "en"
     *   "en" to "es"
     */
    translateViaPivoting(
      first: TranslationModel,
      second: TranslationModel,
      vectorSourceText: VectorString,
      vectorResponseOptions: VectorResponseOptions
    ): VectorResponse;
  }

  /**
   * The actual translation model, which is passed into the `BlockingService` methods.
   */
  export class TranslationModel {}

  /**
   * The models need to be placed in the wasm memory space. This object represents
   * aligned memory that was allocated on the wasm side of things. The memory contents
   * can be set via the getByteArrayView method and the Uint8Array.prototype.set method.
   */
  export class AlignedMemory {
    constructor(size: number, alignment: number);
    size(): number;
    getByteArrayView(): Uint8Array/**
    * The following are the types that are provided by the Bergamot wasm library.
    */;
  }
  
  /**
   * The response from the translation. This definition isn't complete, but just
   * contains a subset of the available methods.
   */
  export class Response {
    getOriginalText(): string;
    getTranslatedText(): string;
  }

  /**
   * The options to configure a translation response.
   */
  export class ResponseOptions {
    // Include the quality estimations.
    qualityScores: boolean;
    // Include the alignments.
    alignment: boolean;
    // Remove HTML tags from text and insert it back into the output.
    html: boolean;
    // Whether to include sentenceMappings or not. Alignments require
    // sentenceMappings and are available irrespective of this option if
    // `alignment=true`.
    sentenceMappings: boolean
  }
}

/**
 * A single language model file.
 */
interface LanguageTranslationModelFile {
  buffer: ArrayBuffer,
}

/**
 * The data required to construct a Bergamot Translation Model.
 */
interface TranslationModelPayload {
  sourceLanguage: string,
  targetLanguage: string,
  languageModelFiles: LanguageTranslationModelFiles,
};

/**
 * The files required to construct a Bergamot Translation Model's aligned memory.
 */
interface LanguageTranslationModelFiles {
  // The machine learning language model.
  model: LanguageTranslationModelFile,
  // The lexical shortlist that limits possible output of the decoder and makes
  // inference faster.
  lex: LanguageTranslationModelFile,
  // A model that can generate a translation quality estimation.
  qualityModel?: LanguageTranslationModelFile,

  // There is either a single vocab file:
  vocab?: LanguageTranslationModelFile,

  // Or there are two:
  srcvocab?: LanguageTranslationModelFile,
  trgvocab?: LanguageTranslationModelFile,
};

/**
 * This is the type that is generated when the models are loaded into wasm aligned memory.
 */
type LanguageTranslationModelFilesAligned = {
  [K in keyof LanguageTranslationModelFiles]: AlignedMemory
};

/**
 * These are the files that are that are necessary to start the translations engine.
 */
interface TranslationsEnginePayload {
  bergamotWasmArrayBuffer: ArrayBuffer,
  translationModelPayloads: TranslationModelPayload[]
}
