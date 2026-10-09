// Shared helpers for building DOM without interpolating untrusted values into
// HTML strings or inline event handlers.
//
// Anything that comes from the server (usernames, group names, file names,
// user agents, task ids, error messages...) must go through SafeDom.el() /
// textContent, or - when an HTML string is unavoidable - SafeDom.escapeHtml().
(function (global) {
  const HTML_ESCAPES = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
    "`": "&#96;",
  };

  // Escape a value for use in HTML text or inside a quoted attribute value.
  // Never use the result inside an inline event handler (onclick="...").
  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value).replace(/[&<>"'`]/g, (ch) => HTML_ESCAPES[ch]);
  }

  function appendChildren(node, children) {
    for (const child of children) {
      if (child === null || child === undefined || child === false) continue;
      if (Array.isArray(child)) {
        appendChildren(node, child);
      } else if (child instanceof Node) {
        node.appendChild(child);
      } else {
        node.appendChild(document.createTextNode(String(child)));
      }
    }
  }

  // Create an element. Supported props:
  //   className, style (cssText string), text, title, dataset {k: v},
  //   attrs {name: value}, on {event: handler}
  // Remaining arguments are children (Node, string, number, arrays, or falsy to skip).
  // Strings are always inserted as text nodes, never parsed as HTML.
  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    const p = props || {};
    if (p.className) node.className = p.className;
    if (p.style) node.style.cssText = p.style;
    if (p.title !== undefined && p.title !== null) node.title = String(p.title);
    if (p.dataset) {
      for (const [key, value] of Object.entries(p.dataset)) {
        node.dataset[key] =
          value === null || value === undefined ? "" : String(value);
      }
    }
    if (p.attrs) {
      for (const [name, value] of Object.entries(p.attrs)) {
        // Event handler attributes would re-introduce inline script.
        if (/^on/i.test(name)) continue;
        if (value === false || value === null || value === undefined) continue;
        node.setAttribute(name, value === true ? "" : String(value));
      }
    }
    if (p.on) {
      for (const [eventName, handler] of Object.entries(p.on)) {
        node.addEventListener(eventName, handler);
      }
    }
    if (p.text !== undefined && p.text !== null)
      node.textContent = String(p.text);
    appendChildren(node, children);
    return node;
  }

  // Replace all children of `parent` with the given nodes.
  function setChildren(parent, ...children) {
    if (!parent) return;
    parent.textContent = "";
    appendChildren(parent, children);
  }

  // Single placeholder table row spanning `colspan` columns.
  function messageRow(colspan, message, color) {
    return el(
      "tr",
      null,
      el("td", {
        attrs: { colspan: colspan },
        style: `text-align:center;color:${color || "#6b7280"};`,
        text: message,
      }),
    );
  }

  // Clamp a server-provided number to a percentage usable in a CSS width.
  function percent(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(100, n));
  }

  global.SafeDom = { escapeHtml, el, setChildren, messageRow, percent };
})(window);
