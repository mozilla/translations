/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for non-pivot translation requests
 * that contain HTML tags within the source text.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "en",
    sourceText: "<b>El perro</b> azul.",
    expectedText: "<b>The</b> blue <b>dog</b>.",
    isHTML: true,
  },
  {
    sourceLanguage: "en",
    targetLanguage: "es",
    sourceText: "<b>The blue</b> dog.",
    expectedText: "<b>El</b> perro <b>azul</b>.",
    isHTML: true,
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "en",
    sourceText: "<b>Le chien</b> bleu.",
    expectedText: "<b>The</b> blue <b>dog</b>.",
    isHTML: true,
  },
  {
    sourceLanguage: "en",
    targetLanguage: "fr",
    sourceText: "<b>The blue</b> dog.",
    expectedText: "<b>Le</b> chien <b>bleu</b>.",
    isHTML: true,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText: "蓝色的<b>狗</b>。",
    expectedText: "The blue <b>dog</b>.",
    isHTML: true,
  },
];

describe("HTML Non-Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
