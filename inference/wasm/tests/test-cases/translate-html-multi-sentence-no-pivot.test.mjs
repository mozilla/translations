/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for non-pivot translation requests
 * that contain HTML tags within the source text, extended to multiple sentences.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "en",
    sourceText:
      "<b>El perro</b> azul. <i>Está corriendo</i> por el parque. Parece muy feliz.",
    expectedText:
      "<b>The</b> blue <b>dog</b>. He's <i>running</i> through the park. He seems very happy.",
    isHTML: true,
  },
  {
    sourceLanguage: "en",
    targetLanguage: "es",
    sourceText:
      "<b>The blue</b> dog. <i>He's running</i> in the park. It looks very happy.",
    expectedText:
      "<b>El</b> perro <b>azul</b>. <i>Está corrido</i> en el parque. Parece muy feliz.",
    isHTML: true,
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "en",
    sourceText:
      "<b>Le chien</b> bleu. <i>Il court</i> dans le parc. Il semble très heureux.",
    expectedText:
      "<b>The</b> blue <b>dog</b>. <i>He runs</i> in the park. He seems very happy.",
    isHTML: true,
  },
  {
    sourceLanguage: "en",
    targetLanguage: "fr",
    sourceText:
      "<b>The blue</b> dog. <i>He's running</i> in the park. It looks very happy.",
    expectedText:
      "<b>Le</b> chien <b>bleu</b>. <i>Il court</i> dans le parc. Il semble très heureux.",
    isHTML: true,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText: "蓝色的<b>狗</b>。<i>它在公园里跑</i>。它看起来很开心。",
    expectedText:
      "The blue <b>dog</b>. <i>It runs in the park</i>. It looks very happy.",
    isHTML: true,
  },
  {
    sourceLanguage: "en",
    targetLanguage: "zh",
    sourceText:
      "<b>The blue</b> dog. <i>He's running</i> in the park. It looks very happy.",
    expectedText:
      // This is temporarily using a Spanish model until we have a trained enzh model.
      // The relevance of the assertion is that whitespace has been omitted between sentences.
      "<b>El</b> perro <b>azul</b>.<i>Está corrido</i> en el parque.Parece muy feliz.",
    isHTML: true,
  },
];

describe("HTML Multi-Sentence Non-Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
