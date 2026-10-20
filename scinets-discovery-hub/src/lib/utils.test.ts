import { describe, it, expect } from "vitest";
import { cn } from "./utils";

describe("cn utility function", () => {
  it("merges class names correctly", () => {
    const result = cn("class-a", "class-b");
    expect(result).toBe("class-a class-b");
  });

  it("handles conditional classes properly", () => {
    const isActive = true;
    const isHidden = false;
    const result = cn("base", isActive && "active", isHidden && "hidden");
    expect(result).toBe("base active");
  });

  it("resolves Tailwind conflicts using twMerge", () => {
    const result = cn("p-4", "p-8");
    expect(result).toBe("p-8");

    const textResult = cn("text-red-500", "text-blue-500");
    expect(textResult).toBe("text-blue-500");
  });

  it("handles arrays and falsy values", () => {
    const result = cn(["item-1", null, undefined, false, "item-2"]);
    expect(result).toBe("item-1 item-2");
  });
});
