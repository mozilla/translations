/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for non-pivot translation requests
 * that contain plain text with multiple sentences, without HTML tags.
 */
const testCases = [
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText: "我走近魔法门。“我不会给你开门的。”门内传来一个声音。",
    expectedText: `I approached the magic door. "I'm not going to open the door for you." There was a sound inside the door.`,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText: "我终于到达了魔法门！“我不会给你开门的。”门内传来一个声音。",
    expectedText: `I finally got to the magic door! "I'm not going to open the door for you." There was a sound inside the door.`,
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText: "你以为这就是魔法门吗？“我不会给你开的。”门内传来一个声音。",
    expectedText: `Do you think this is the magic door? "I'm not going to give you." There was a sound inside the door.`,
  },
];

describe("Plain-Text Multi-Sentence Non-Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
