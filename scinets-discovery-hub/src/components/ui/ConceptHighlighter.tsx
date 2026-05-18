import React, { useMemo } from "react";

interface ConceptHighlighterProps {
  text: string;
  concepts: string[];
  className?: string;
}

export function ConceptHighlighter({ text, concepts, className }: ConceptHighlighterProps) {
  const parts = useMemo(() => {
    if (!text || !concepts || concepts.length === 0) {
      return [{ text, isConcept: false, concept: null as string | null }];
    }

    const validConcepts = concepts.filter((c) => c && c.length > 2);
    if (validConcepts.length === 0) {
      return [{ text, isConcept: false, concept: null as string | null }];
    }

    validConcepts.sort((a, b) => b.length - a.length);

    const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

    const pattern = new RegExp(
      `\\b(${validConcepts.map((c) => {
        const escaped = escapeRegExp(c);
        if (!c.includes(" ") && !c.toLowerCase().endsWith("s")) {
          return `${escaped}(?:s|es)?`;
        }
        return escaped;
      }).join("|")})\\b`,
      "gi"
    );

    const result: Array<{ text: string; isConcept: boolean; concept: string | null }> = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = pattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        result.push({
          text: text.substring(lastIndex, match.index),
          isConcept: false,
          concept: null,
        });
      }

      result.push({
        text: match[0],
        isConcept: true,
        concept: match[0],
      });

      lastIndex = pattern.lastIndex;
    }

    if (lastIndex < text.length) {
      result.push({
        text: text.substring(lastIndex),
        isConcept: false,
        concept: null,
      });
    }

    return result;
  }, [text, concepts]);

  return (
    <span className={className}>
      {parts.map((part, i) => {
        if (part.isConcept && part.concept) {
          const wikiUrl = `https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(part.concept)}`;
          return (
            <a
              key={i}
              href={wikiUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary font-medium underline decoration-primary/30 hover:decoration-primary transition-colors duration-200"
              title={`Search Wikipedia for "${part.concept}"`}
            >
              {part.text}
            </a>
          );
        }
        return <React.Fragment key={i}>{part.text}</React.Fragment>;
      })}
    </span>
  );
}
