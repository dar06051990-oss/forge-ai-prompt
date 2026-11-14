/** Forge AI Prompt: correctly centered, viewport-level preset editor.
 * The backend callbacks remain Gradio-native; moving only the DOM container
 * avoids Forge's nested extension panels clipping position:fixed elements.
 */
(() => {
    "use strict";
    if (window.__forgeAiPromptModal028) return;
    window.__forgeAiPromptModal028 = true;

    const SUFFIXES = ["t2i", "i2i"];
    const states = Object.create(null);
    let active = null;
    let priorOverflow = null;
    let backdrop = null;

    function appRoot() {
        try {
            if (typeof gradioApp === "function") return gradioApp() || document;
        } catch (_) {}
        return document;
    }

    function lookup(id) {
        const root = appRoot();
        return (typeof root.getElementById === "function" ? root.getElementById(id) : null)
            || root.querySelector?.(`#${id}`)
            || document.getElementById(id);
    }

    function portalHost() {
        const root = appRoot();
        // Keep Gradio styles intact if the app is using Shadow DOM.
        return root instanceof ShadowRoot ? root : document.body;
    }

    function mount(suffix) {
        if (active !== suffix) return;
        const node = lookup(`fap_preset_editor_${suffix}`);
        if (!node) return;
        let state = states[suffix];
        if (!state || state.node !== node) {
            const anchor = document.createComment(`fap-preset-editor-${suffix}`);
            node.parentNode?.insertBefore(anchor, node);
            state = { node, anchor };
            states[suffix] = state;
        }
        if (node.parentNode !== portalHost()) {
            portalHost().appendChild(node);
        }
        node.classList.remove("fap-editor-closing");
        node.classList.add("fap-editor-portaled");
        node.setAttribute("role", "dialog");
        node.setAttribute("aria-modal", "true");
        node.setAttribute("aria-label", "System prompt presets");

        if (!backdrop) {
            backdrop = document.createElement("div");
            backdrop.className = "fap-preset-backdrop";
            backdrop.setAttribute("aria-hidden", "true");
        }
        if (backdrop.parentNode !== portalHost()) {
            portalHost().appendChild(backdrop);
        }
        // Keep modal above the backdrop in normal stacking order, too.
        if (node.parentNode === backdrop.parentNode && node.nextSibling !== null) {
            node.parentNode.appendChild(node);
        }
    }

    function open(suffix) {
        if (active && active !== suffix) close(active);
        active = suffix;
        if (!priorOverflow) {
            priorOverflow = {
                body: document.body.style.overflow,
                html: document.documentElement.style.overflow,
            };
        }
        document.body.style.overflow = "hidden";
        document.documentElement.style.overflow = "hidden";
        mount(suffix);
        // A group with visible=False can be inserted only after Gradio's
        // asynchronous callback updates it; try after the update as well.
        [0, 80, 220, 450].forEach(ms => setTimeout(() => mount(suffix), ms));
    }

    function close(suffix) {
        const state = states[suffix];
        if (state?.node) {
            state.node.classList.add("fap-editor-closing");
            state.node.classList.remove("fap-editor-portaled");
            state.node.removeAttribute("aria-modal");
            if (state.anchor?.parentNode) {
                state.anchor.parentNode.insertBefore(state.node, state.anchor);
                state.anchor.remove();
            }
            delete states[suffix];
        }
        if (active !== suffix) return;
        active = null;
        if (backdrop) backdrop.remove();
        if (priorOverflow) {
            document.body.style.overflow = priorOverflow.body;
            document.documentElement.style.overflow = priorOverflow.html;
            priorOverflow = null;
        }
    }

    function suffixFromEvent(event, prefix) {
        const path = event.composedPath?.() || [event.target];
        for (const el of path) {
            if (!el?.id || typeof el.id !== "string") continue;
            if (el.id.startsWith(prefix)) {
                const suffix = el.id.slice(prefix.length);
                if (SUFFIXES.includes(suffix)) return suffix;
            }
        }
        return null;
    }

    function onClick(event) {
        if (event.__fapModalProcessed) return;
        let suffix = suffixFromEvent(event, "fap_preset_open_");
        if (suffix) {
            event.__fapModalProcessed = true;
            open(suffix);
            return;
        }
        suffix = suffixFromEvent(event, "fap_preset_close_");
        if (suffix) {
            event.__fapModalProcessed = true;
            close(suffix);
        }
    }

    function onKeydown(event) {
        if (event.key !== "Escape" || !active) return;
        const closeButton = lookup(`fap_preset_close_${active}`);
        if (!closeButton) return;
        event.preventDefault();
        closeButton.click();
    }

    const listening = new WeakSet();
    function installListeners() {
        for (const root of [document, appRoot()]) {
            if (listening.has(root)) continue;
            root.addEventListener("click", onClick, true);
            root.addEventListener("keydown", onKeydown, true);
            listening.add(root);
        }
    }

    installListeners();
    document.addEventListener("DOMContentLoaded", installListeners, { once: true });
    if (typeof onUiLoaded === "function") onUiLoaded(installListeners);
    // The Gradio root or hidden editor can be mounted after the click's
    // asynchronous Python callback. Observe its subtree, not just <html>.
    // mount() is idempotent, so our own portal moves settle after one pass.
    const watched = new WeakSet();
    function watchGradio() {
        const root = appRoot();
        if (watched.has(root)) return;
        const observer = new MutationObserver(() => {
            if (active) mount(active);
            installListeners();
        });
        observer.observe(root, { childList: true, subtree: true });
        watched.add(root);
    }
    watchGradio();
    // Watch for a late-created gradio-app node / shadow root.
    const bootObserver = new MutationObserver(() => {
        installListeners();
        watchGradio();
    });
    bootObserver.observe(document.documentElement, { childList: true, subtree: false });
})();
