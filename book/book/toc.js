// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">Getting Started</li><li class="chapter-item expanded "><a href="getting-started/installation.html"><strong aria-hidden="true">1.</strong> Installation</a></li><li class="chapter-item expanded "><a href="getting-started/quick-start.html"><strong aria-hidden="true">2.</strong> Quick Start</a></li><li class="chapter-item expanded "><a href="getting-started/structure.html"><strong aria-hidden="true">3.</strong> Project Structure</a></li><li class="chapter-item expanded affix "><li class="part-title">Core Concepts</li><li class="chapter-item expanded "><a href="concepts/apr-format.html"><strong aria-hidden="true">4.</strong> The APR Format</a></li><li class="chapter-item expanded "><a href="concepts/bundling.html"><strong aria-hidden="true">5.</strong> Model Bundling</a></li><li class="chapter-item expanded "><a href="concepts/conversion.html"><strong aria-hidden="true">6.</strong> Format Conversion</a></li><li class="chapter-item expanded "><a href="concepts/zero-copy.html"><strong aria-hidden="true">7.</strong> Zero-Copy Loading</a></li><li class="chapter-item expanded affix "><li class="part-title">Recipes</li><li class="chapter-item expanded "><a href="recipes/bundle-static.html"><strong aria-hidden="true">8.</strong> Bundle a Static Model</a></li><li class="chapter-item expanded "><a href="recipes/bundle-quantized.html"><strong aria-hidden="true">9.</strong> Bundle with Quantization</a></li><li class="chapter-item expanded "><a href="recipes/encrypt-model.html"><strong aria-hidden="true">10.</strong> Encrypt a Model</a></li><li class="chapter-item expanded "><a href="recipes/convert-safetensors.html"><strong aria-hidden="true">11.</strong> Convert from SafeTensors</a></li><li class="chapter-item expanded "><a href="recipes/convert-gguf.html"><strong aria-hidden="true">12.</strong> Convert from GGUF</a></li><li class="chapter-item expanded "><a href="recipes/export-gguf.html"><strong aria-hidden="true">13.</strong> Export to GGUF</a></li><li class="chapter-item expanded "><a href="recipes/simd-acceleration.html"><strong aria-hidden="true">14.</strong> SIMD Acceleration</a></li><li class="chapter-item expanded affix "><li class="part-title">CLI Tools</li><li class="chapter-item expanded "><a href="cli/apr-info.html"><strong aria-hidden="true">15.</strong> apr-info</a></li><li class="chapter-item expanded "><a href="cli/apr-bench.html"><strong aria-hidden="true">16.</strong> apr-bench</a></li><li class="chapter-item expanded affix "><li class="part-title">Advanced Topics</li><li class="chapter-item expanded "><a href="advanced/wasm.html"><strong aria-hidden="true">17.</strong> WASM Deployment</a></li><li class="chapter-item expanded "><a href="advanced/custom-models.html"><strong aria-hidden="true">18.</strong> Custom Model Types</a></li><li class="chapter-item expanded "><a href="advanced/performance.html"><strong aria-hidden="true">19.</strong> Performance Optimization</a></li><li class="chapter-item expanded affix "><li class="part-title">Reference</li><li class="chapter-item expanded "><a href="reference/api.html"><strong aria-hidden="true">20.</strong> API Documentation</a></li><li class="chapter-item expanded "><a href="reference/errors.html"><strong aria-hidden="true">21.</strong> Error Handling</a></li><li class="chapter-item expanded "><a href="reference/features.html"><strong aria-hidden="true">22.</strong> Feature Flags</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
