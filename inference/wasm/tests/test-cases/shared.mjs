/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { it, expect } from "vitest";
import { TranslationsEngine } from "./engine/translations-engine.mjs";

/**
 * Runs a translation test, constructing a Translations Engine for the given
 * sourceLanguage and targetLanguage, then asserting that the translation of
 * the sourceText matches the expectedText.
 *
 * @param {Object} params - The parameters for the test.
 * @param {string} params.sourceLanguage - The source language code.
 * @param {string} params.targetLanguage - The target language code.
 * @param {string} params.sourceText - The text to translate.
 * @param {string} params.expectedText - The expected translated text.
 * @param {boolean} params.isHTML - Whether the text to translate contains HTML tags.
 */
export function runTranslationTest({
  sourceLanguage,
  targetLanguage,
  sourceText,
  expectedText,
  isHTML = false,
}) {
  it(`(${sourceLanguage} -> ${targetLanguage}): Translate "${sourceText.replaceAll("\n", " ")}"`, async () => {
    const translator = new TranslationsEngine(sourceLanguage, targetLanguage);

    const translatedText = await translator.translate(sourceText, isHTML);

    expect(translatedText).toBe(expectedText);

    translator.terminate();
  });
}
