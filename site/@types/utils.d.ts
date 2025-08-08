/**
 * The interface used by the createElements utility.
 */
export interface CreateElementOptions {
  style: Partial<CSSStyleDeclaration>,
  attrs: Record<string, string | number | null>
  parent: Element,
  children: Node | string | number | Array<string | Node | number> | [],
  href: string,
  className: string,
  title: string,
  onClick:(this: HTMLElement, event: MouseEvent) => unknown
}
