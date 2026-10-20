import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "./button";

describe("Button component", () => {
  it("renders with default text and classes", () => {
    render(<Button>Click me</Button>);
    const button = screen.getByRole("button", { name: /click me/i });
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass("bg-primary");
  });

  it("applies variant classes correctly", () => {
    const { rerender } = render(<Button variant="destructive">Delete</Button>);
    let button = screen.getByRole("button", { name: /delete/i });
    expect(button).toHaveClass("bg-destructive");

    rerender(<Button variant="outline">Outline</Button>);
    button = screen.getByRole("button", { name: /outline/i });
    expect(button).toHaveClass("border");

    rerender(<Button variant="secondary">Secondary</Button>);
    button = screen.getByRole("button", { name: /secondary/i });
    expect(button).toHaveClass("bg-secondary");
  });

  it("applies size classes correctly", () => {
    const { rerender } = render(<Button size="sm">Small</Button>);
    let button = screen.getByRole("button", { name: /small/i });
    expect(button).toHaveClass("h-9");

    rerender(<Button size="lg">Large</Button>);
    button = screen.getByRole("button", { name: /large/i });
    expect(button).toHaveClass("h-11");
  });

  it("handles click events", async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(<Button onClick={handleClick}>Submit</Button>);
    const button = screen.getByRole("button", { name: /submit/i });

    await user.click(button);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it("does not fire click when disabled", async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(<Button disabled onClick={handleClick}>Disabled</Button>);
    const button = screen.getByRole("button", { name: /disabled/i });

    expect(button).toBeDisabled();
    await user.click(button);
    expect(handleClick).not.toHaveBeenCalled();
  });
});
