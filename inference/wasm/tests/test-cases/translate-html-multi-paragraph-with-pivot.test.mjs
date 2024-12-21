/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for pivot translation requests
 * that contain HTML tags within the source text, extended to multiple paragraphs.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "fr",
    sourceText:
      "<b>El perro</b> azul. <u><i>Está corriendo</i> por el parque.\n\nParece muy feliz.</u> Juega con otros perros.",
    expectedText:
      "<b>Le chien</b> bleu. <u><i>Il court</i> à travers le parc.\n\nIl a l'air très heureux.</u> Jouez avec d'autres chiens.",
    isHTML: true,
  },
  {
    sourceLanguage: "es",
    targetLanguage: "zh",
    sourceText:
      "<b>El perro</b> azul. <u><i>Está corriendo</i> por el parque.\n\nParece muy feliz.</u> Juega con otros perros.",
    expectedText:
      // This is temporarily using a Spanish model until we have a trained enzh model.
      // The relevance of the assertion is that whitespace has been omitted between sentences.
      "<b>El perro</b> azul.<u><i>Corre</i> por el parque.\n\nParece muy feliz.</u>Juega con otros perros.",
    isHTML: true,
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "es",
    sourceText:
      "<b>Le chien</b> bleu. <u><i>Il court</i> dans le parc.\n\nIl semble très heureux.</u> Il joue avec d'autres chiens.",
    expectedText:
      "<b>El perro</b> azul. <u><i>Corre</i> en el parque.\n\nParece muy feliz.</u> Juega con otros perros.",
    isHTML: true,
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "zh",
    sourceText:
      "<b>Le chien</b> bleu. <u><i>Il court</i> dans le parc.\n\nIl semble très heureux.</u> Il joue avec d'autres chiens.",
    expectedText:
      // This is temporarily using a Spanish model until we have a trained enzh model.
      // The relevance of the assertion is that whitespace has been omitted between sentences.
      "<b>El perro</b> azul.<u><i>Corre</i> en el parque.\n\nParece muy feliz.</u>Juega con otros perros.",
    isHTML: true,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "es",
    sourceText:
      "<b>蓝色的狗。</b><u><i>它在公园里跑。</i>\n\n它看起来很开心。</u>它和其他狗一起玩。",
    expectedText:
      "<b>El perro azul. </b><u><i>Corre en el parque.</i>\n\nParece muy feliz. </u>Está jugando con otros perros.",
    isHTML: true,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "fr",
    sourceText:
      "<b>蓝色的狗。</b><u><i>它在公园里跑。</i>\n\n它看起来很开心。</u>它和其他狗一起玩。",
    expectedText:
      "<b>Le chien bleu. </b><u><i>Il court dans le parc.</i>\n\nIl semble très heureux. </u>Il joue avec d'autres chiens.",
    isHTML: true,
  },
];

describe("HTML Multi-Paragraph Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
