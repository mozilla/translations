/** @type {Terminology} */
export const terminology = {
  adequacy: {
    description:
      "Measures how well the meaning of the source sentence is preserved in the translation.",
    scales: {
      5: "Complete meaning preserved",
      4: "Minor meaning issues",
      3: "Partial meaning loss",
      2: "Major meaning errors",
      1: "Meaning not preserved",
    },
  },
  fluency: {
    description:
      "Assesses the grammatical correctness and naturalness of the translation.",
    scales: {
      5: "Fluent and natural",
      4: "Minor grammatical issues",
      3: "Awkward but understandable",
      2: "Difficult to understand",
      1: "Ungrammatical or nonsensical",
    },
  },
  terminology: {
    description:
      "Evaluates the correct and consistent use of domain-specific or technical terms.",
    scales: {
      5: "Correct and consistent use",
      4: "Minor inconsistency or misuse",
      3: "Some incorrect or inconsistent terms",
      2: "Frequent term issues",
      1: "Terminology mostly incorrect",
    },
  },
  hallucination: {
    description:
      "Checks whether the translation introduces unsupported or invented content.",
    scales: {
      5: "No hallucinations",
      4: "Minor unsupported additions",
      3: "Some invented content",
      2: "Frequent hallucinations",
      1: "Hallucinated most content",
    },
  },
  punctuation: {
    description:
      "Evaluates the appropriateness and correctness of punctuation in the translation.",
    scales: {
      5: "Punctuation is appropriate and matches expectations",
      4: "Minor punctuation issues (e.g., missing stylistic marks)",
      3: "Some punctuation errors or inconsistencies",
      2: "Frequent punctuation problems",
      1: "Punctuation is largely incorrect or absent",
    },
  },
};
