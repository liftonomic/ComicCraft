/** @type {import('tailwindcss').Config} */
module.exports = {
  // Only scan the real markup so the generated CSS stays small.
  content: ["./index.html"],
  theme: {
    extend: {},
  },
  plugins: [],
  // The project uses custom classes such as `hidden-force`; keep preflight
  // enabled so the offline build matches the previous CDN behaviour exactly.
  corePlugins: {
    preflight: true,
  },
};
