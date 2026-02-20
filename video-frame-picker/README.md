# Video Frame Picker (One-Page Website)

This is a free, static one-page tool that:
- accepts video upload in browser-supported formats,
- extracts and displays the last 5 frames,
- downloads clicked frame as PNG.

## Local run (no coding tools required)
1. Open `index.html` directly in your browser.
2. Upload a video.
3. Click any shown frame to download.

## Make it live (Netlify - easiest)
1. Create account at https://www.netlify.com/.
2. Drag-drop the `video-frame-picker` folder into Netlify deploy page.
3. Netlify gives a live URL instantly.
4. Add your custom domain (optional but recommended for SEO).

## SEO setup before production
1. In `index.html`, replace all `https://example.com/` with your real domain.
2. In `robots.txt`, replace sitemap URL with your real domain.
3. In `sitemap.xml`, replace `<loc>` with your real domain.
4. Submit sitemap to Google Search Console.

## Browser note
Browsers do not decode every codec. For best compatibility, use MP4 (H.264) or WebM.

## Test cases
Use the checklist in `test-cases.md`.
