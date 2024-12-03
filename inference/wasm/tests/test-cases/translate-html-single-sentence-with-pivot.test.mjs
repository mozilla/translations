/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for pivot translation requests
 * that contain HTML tags within the source text.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "fr",
    sourceText: "<b>El perro</b> azul.",
    expectedText: "<b>Le chien</b> bleu.",
    isHTML: true,
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "es",
    sourceText: "<b>Le chien</b> bleu.",
    expectedText: "<b>El perro</b> azul.",
    isHTML: true,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "es",
    sourceText: "蓝色的<b>狗</b>。",
    expectedText: "El <b>perro</b> azul.",
    isHTML: true,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "fr",
    sourceText: "蓝色的<b>狗</b>。",
    expectedText: "Le <b>chien</b> bleu.",
    isHTML: true,
  },
];

describe("HTML Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
