/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for non-pivot translation requests
 * that contain HTML tags within the source text, extended to multiple paragraphs.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "en",
    sourceText:
      "<b>El perro azul.</b> <u><i>Está corrindo por el parque.\n\nParece muy feliz.</i></u> Juega con otros perros.",
    expectedText:
      "<b>The blue dog.</b> <u><i>He's running through the park.\n\nHe seems very happy.</i></u> Play with other dogs.",
    isHTML: true,
  },
  {
    sourceLanguage: "en",
    targetLanguage: "es",
    sourceText:
      "<b>The blue dog.</b> <u><i>He is running through the park.\n\nHe seems very happy.</i></u> He plays with other dogs.",
    expectedText:
      "<b>El perro azul.</b> <u><i>Está corrindo por el parque.\n\nParece muy feliz.</i></u> Juega con otros perros.",
    isHTML: true,
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "en",
    sourceText:
      "<b>Le chien bleu.</b> <u><i>Il court dans le parc.\n\nIl semble très heureux.</i></u> Il joue avec d'autres chiens.",
    expectedText:
      "<b>The blue dog.</b> <u><i>He runs in the park.\n\nHe seems very happy.</i></u> He plays with other dogs.",
    isHTML: true,
  },
  {
    sourceLanguage: "en",
    targetLanguage: "fr",
    sourceText:
      "<b>The blue dog.</b> <u><i>He is running in the park.\n\nHe seems very happy.</i></u> He plays with other dogs.",
    expectedText:
      "<b>Le chien bleu.</b> <u><i>Il court dans le parc.\n\nIl a l'air très heureux.</i></u> Il joue avec d'autres chiens.",
    isHTML: true,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText:
      "<b>蓝色的狗。</b><u><i>它在公园里跑。\n\n它看起来很开心。</i></u>它和其他狗一起玩。",
    expectedText:
      "<b>The blue dog. </b><u><i>It runs in the park.\n\nIt looks very happy. </i></u>It's playing with other dogs.",
    isHTML: true,
  },
  {
    sourceLanguage: "en",
    targetLanguage: "zh",
    sourceText:
      "<b>The blue dog.</b> <u><i>He is running through the park.\n\nHe seems very happy.</i></u> He plays with other dogs.",
    expectedText:
      // This is temporarily using a Spanish model until we have a trained enzh model.
      // The relevance of the assertion is that whitespace has been omitted between sentences.
      "<b>El perro azul.</b><u><i>Está corrindo por el parque.\n\nParece muy feliz.</i></u>Juega con otros perros.",
    isHTML: true,
  },
];

describe("HTML Multi-Paragraph Non-Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
