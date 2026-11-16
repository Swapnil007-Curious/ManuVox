// The offline gloss -> sentence joiner (the "(basic)" fallback) and the sign lookup it uses.

import { describe, expect, it } from "vitest";
import { basicSentence } from "../../src/ui/sentence";
import { dictionaryUrl, glossOf, searchSigns, SIGNS, wordsOf } from "../../src/ui/signs";

describe("basicSentence", () => {
  it("lower-cases, capitalises the first word and ends with a full stop", () => {
    expect(basicSentence(["mom", "drink", "water"])).toBe("Mom drink water.");
  });

  it("puts a comma after a leading greeting", () => {
    expect(basicSentence(["hello", "mom", "drink", "water"])).toBe("Hello, mom drink water.");
    expect(basicSentence(["no", "milk"])).toBe("No, milk.");
  });

  it("splits compound glosses into words", () => {
    expect(basicSentence(["thankyou"])).toBe("Thank you.");
    expect(basicSentence(["haveto", "go", "home"])).toBe("Have to go home.");
    expect(basicSentence(["icecream", "frenchfries"])).toBe("Ice cream french fries.");
    expect(basicSentence(["minemy", "dog"])).toBe("My dog.");
  });

  it("asks a question when a question word is present", () => {
    expect(basicSentence(["where", "dad"])).toBe("Where dad?");
    expect(basicSentence(["hello", "who", "that"])).toBe("Hello, who that?");
  });

  it("keeps TV in capitals and greets with an exclamation when alone", () => {
    expect(basicSentence(["TV", "now"])).toBe("TV now.");
    expect(basicSentence(["dad", "TV"])).toBe("Dad TV.");
    expect(basicSentence(["hello"])).toBe("Hello!");
    expect(basicSentence(["yes"])).toBe("Yes.");
  });

  it("adds nothing and drops nothing", () => {
    expect(basicSentence([])).toBe("");
    expect(basicSentence(["water", "water"])).toBe("Water water.");
  });
});

describe("signs", () => {
  it("has a gloss, words, slug and description for each of the 250 labels", () => {
    expect(SIGNS).toHaveLength(250);
    for (const sign of SIGNS) {
      expect(sign.gloss, sign.label).toMatch(/^[A-Z][A-Z/-]*$|^TV$/);
      expect(sign.words.length, sign.label).toBeGreaterThan(0);
      expect(sign.description.length, sign.label).toBeGreaterThan(10);
    }
  });

  it("maps compound labels to readable glosses, words and dictionary pages", () => {
    expect(glossOf("thankyou")).toBe("THANK-YOU");
    expect(wordsOf("callonphone")).toBe("call");
    expect(dictionaryUrl("thankyou")).toBe("https://www.signasl.org/sign/thank-you");
    expect(dictionaryUrl("owie")).toBe("https://www.signasl.org/sign/hurt");
    expect(glossOf("not-a-label")).toBe("NOT-A-LABEL");
  });

  it("searches names first, then descriptions", () => {
    const names = searchSigns("water").map((sign) => sign.label);
    expect(names[0]).toBe("water");
    expect(searchSigns("thank you")[0]?.label).toBe("thankyou");
    expect(searchSigns("chin").map((sign) => sign.label)).toContain("mom"); // description match
    expect(searchSigns("zzzz")).toEqual([]);
    expect(searchSigns("   ")).toHaveLength(250);
  });

  it("lists every sign alphabetically when the query is empty (not in model label order)", () => {
    const glosses = searchSigns("").map((sign) => sign.gloss);
    expect(glosses[0]).toBe("AFTER"); // the label order starts with "TV"
    expect(glosses).toEqual([...glosses].sort((a, b) => a.localeCompare(b, "en")));
  });
});
