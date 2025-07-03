// @ts-check

/**
 * @import { CreateElementOptions } from "./@types/utils"
 */

/**
 * @param {any} any
 * @returns {any}
 */
export function asAny(any) {
  return any;
}

/**
 * @param {URLSearchParams} urlParams
 */
export function replaceLocation(urlParams) {
  const url = new URL(window.location.href);
  const newLocation = `${url.origin}${url.pathname}?${urlParams}`;
  history.replaceState(null, "", newLocation);
}

/**
 * @param {URLSearchParams} urlParams
 */
export function changeLocation(urlParams) {
  const url = new URL(window.location.href);
  const newLocation = `${url.origin}${url.pathname}?${urlParams}`;

  // @ts-ignore
  window.location = newLocation;
}

/**
 * @param {string} key
 * @param {any} value
 */
export function exposeAsGlobal(key, value) {
  console.log(key, value);
  asAny(window)[key] = value;
}

/**
 * Gets an element and throws if it doesn't exists. The className provided to specialize
 * the type. Note the `any` coercion for the HTMLElement is working around an issue
 * where TypeScript complains about the types. This workaround makes it so that the
 * types are correctly inferred, and there are no runtime errors.
 *
 * @template {HTMLElement} T
 *
 * @param {string} id
 * @param {{ new (): T }} className
 * @returns {T}
 */
export function getElement(id, className = /** @type {any} */ (HTMLElement)) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error("Could not find element by id: " + id);
  }
  if (!(element instanceof className)) {
    throw new Error(
      `Selected element #${id} was not an instance of ${className.name}`
    );
  }
  return element;
}

/**
 * Helper to create a table row, and add TD elements.
 *
 * @param {HTMLElement} tbody
 * @param {Element?} [insertBefore]
 */
export function createTableRow(tbody, insertBefore) {
  const tr = document.createElement("tr");
  tbody.insertBefore(tr, insertBefore ?? null);

  return {
    tr,
    /**
     * @param {string | Element} [textOrEl]
     * @returns {HTMLTableCellElement}
     */
    createTD(textOrEl = "") {
      const el = document.createElement("td");
      if (typeof textOrEl === "string") {
        el.innerText = textOrEl;
      } else {
        el.appendChild(textOrEl);
      }
      tr.appendChild(el);
      return el;
    },
  };
}

/**
 * Helper to create an <a href> tag.
 *
 * @param {string} text
 * @param {string} [href]
 */
export function createLink(text, href) {
  const a = document.createElement("a");
  if (href) {
    a.href = href;
  }
  a.innerText = text;
  return a;
}

/**
 * Helper to create a button with an action.
 *
 * @param {string | Element} textOrEl
 * @param {(this: HTMLButtonElement, event: MouseEvent) => unknown} callback
 */
export function createButton(textOrEl, callback) {
  const button = document.createElement("button");
  button.addEventListener("click", callback);
  if (typeof textOrEl === "string") {
    button.innerText = textOrEl;
  } else {
    button.appendChild(textOrEl);
  }
  return button;
}

/**
 * Formats a number of bytes into a human-readable string.
 *
 * @param {number} bytes
 * @param {number} [decimals]
 * @returns {string}
 */
export function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return "0 B";

  const k = 1000;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

/**
 * @typedef {object} SearchFilters
 * @property {string} key
 * @property {string} value
 * @property {boolean} negated
 */

/**
 * AI-generated search query parser, manually tweaked.
 *
 *   > Write a JS parser for the following search syntax:
 *   >
 *   > name:search-term date:>2025-01-02 -language:french
 *   >
 *   > term1 term2
 *   >
 *   > "quoted term"
 *   >
 *   > name:"quoted term" date:<2025-01-12
 *
 * @param {string} query
 * @returns {{ filters: SearchFilters[], terms: string[] }}
 */
export function parseSearchQuery(query) {
  const fieldPattern = /(?:^|\s)(-?)(\w+):(\"[^\"]+\"|[^\s]+)/g;
  const unstructuredPattern = /(?:^|\s)(\"[^\"]+\"|\S+)/g;

  let match;
  /** @type {SearchFilters[]} */
  const filters = [];
  /** @type {string[]} */
  const terms = [];

  const seenIndices = new Set();

  // Extract field-based filters
  while ((match = fieldPattern.exec(query)) !== null) {
    const [, negation, key, rawValue] = match;
    const value = rawValue.replace(/^\"|\"$/g, "").trim();

    if (value) {
      filters.push({
        key: key.toLowerCase(),
        value: value.toLowerCase(),
        negated: !!negation,
      });
    }
    seenIndices.add(match.index);
  }

  // Extract unstructured search terms
  while ((match = unstructuredPattern.exec(query)) !== null) {
    if (!seenIndices.has(match.index)) {
      const value = match[1].replace(/^\"|\"$/g, "");
      if (value.trim()) {
        terms.push(value.toLowerCase());
      }
    }
  }

  return { filters, terms };
}

/**
 * @param {URLSearchParams} urlParams
 */
export function pushLocation(urlParams) {
  const url = new URL(window.location.href);
  const newLocation = `${url.origin}${url.pathname}?${urlParams}`;
  history.pushState(null, "", newLocation);
}

/**
 * @param {any} object
 */
export function jsonToYAML(object, indent = 0) {
  const spaces = "  ".repeat(indent);
  let yaml = "";

  for (const [key, value] of Object.entries(object)) {
    const formattedKey = /^[a-zA-Z0-9_-]+$/.test(key) ? key : `'${key}'`;

    if (value === null) {
      yaml += `${spaces}${formattedKey}: null\n`;
    } else if (typeof value === "boolean" || typeof value === "number") {
      yaml += `${spaces}${formattedKey}: ${value}\n`;
    } else if (typeof value === "string") {
      yaml += `${spaces}${formattedKey}: ${
        value.includes(":") || value.includes("\n")
          ? `|\n${spaces}  ` + value.replace(/\n/g, `\n${spaces}  `)
          : value
      }\n`;
    } else if (Array.isArray(value)) {
      if (value.length === 0) {
        yaml += `${spaces}${formattedKey}: []\n`;
      } else {
        yaml += `${spaces}${formattedKey}:\n`;
        for (const item of value) {
          yaml += `${spaces}  - ${
            typeof item === "object"
              ? "\n" + jsonToYAML(item, indent + 2)
              : item
          }\n`;
        }
      }
    } else if (typeof value === "object") {
      yaml += `${spaces}${formattedKey}:\n` + jsonToYAML(value, indent + 1);
    }
  }

  return yaml;
}

/**
 * A type-only check that a type is "never"
 * @param {never} never
 */
export function isNever(never) {}

/**
 * A utility function to make it easier to create HTML elements declaratively.
 *
 * @template {keyof HTMLElementTagNameMap} T
 *
 * @param {T} tagName
 * @param {Partial<CreateElementOptions>} [options]
 */
export function createElement(tagName, options) {
  const element = document.createElement(tagName);
  if (options) {
    const { style, parent, children, href, className, title, onClick } =
      options;
    if (style) {
      Object.assign(element.style, style);
    }
    if (href !== undefined) {
      if (element instanceof HTMLAnchorElement) {
        element.href = href;
      } else {
        throw new Error("An href was provided for a non-anchor element.");
      }
    }
    if (typeof children === "string") {
      element.innerText = children;
    } else if (Array.isArray(children)) {
      for (const child of children) {
        if (typeof child === "string") {
          element.appendChild(new Text(child));
        } else {
          element.appendChild(child);
        }
      }
    } else if (children instanceof Node) {
      element.appendChild(children);
    } else if (children) {
      // Ensure we've handled all of the cases.
      isNever(children);
    }

    if (className) {
      element.className = className;
    }

    if (title) {
      element.title = title;
    }

    if (onClick) {
      if (element instanceof HTMLButtonElement) {
        element.addEventListener("click", onClick);
      } else {
        throw new Error(
          "The createElement util needs support for this onClick handler"
        );
      }
    }

    // Append it last to avoid unnecessary jank.
    if (parent) {
      parent.appendChild(element);
    }
  }
  return element;
}

/**
 * A subset of supported tag names, feel free to add more tag names.
 */
const tagNames = [
  /** @type {const} */ ("a"),
  /** @type {const} */ ("br"),
  /** @type {const} */ ("button"),
  /** @type {const} */ ("div"),
  /** @type {const} */ ("h1"),
  /** @type {const} */ ("h2"),
  /** @type {const} */ ("h3"),
  /** @type {const} */ ("h4"),
  /** @type {const} */ ("li"),
  /** @type {const} */ ("p"),
  /** @type {const} */ ("pre"),
  /** @type {const} */ ("span"),
  /** @type {const} */ ("table"),
  /** @type {const} */ ("tbody"),
  /** @type {const} */ ("thead"),
  /** @type {const} */ ("th"),
  /** @type {const} */ ("td"),
  /** @type {const} */ ("tr"),
  /** @type {const} */ ("ul"),
];

/**
 * @typedef {(typeof tagNames)[number]} TagNames
 */

/**
 * @typedef {typeof createElement} CreateElement
 */

/**
 * Exports the createElement interface in a convenient partially applied interface that
 * can autocomplete. To support additional tag names, add the tag to the tagNames list.
 *
 * The simplified type for the interface is (the HTMLElement is specialized to the specific type).
 *
 * (options?: Partial<CreateElementOptions>) => HTMLElement
 *
 * @see {CreateElementOptions}
 *
 * Instead of:
 *
 *   const coolDiv = createElement("div", {
 *     className: "cool-div",
 *   });
 *
 * You can write:
 *
 *   const coolDiv = create.div({
 *     className: "cool-div",
 *   });
 *
 * @type {{ [K in TagNames]: (options?: Partial<CreateElementOptions>) => HTMLElementTagNameMap[K]; }}
 */
export const create = /** @type {any} */ ({});

// Create the partially applied createElement functions.
for (const tagName of tagNames) {
  /**
   * @type {(options?: Partial<CreateElementOptions>) => any}
   */
  create[tagName] = (options) => createElement(tagName, options);
}
