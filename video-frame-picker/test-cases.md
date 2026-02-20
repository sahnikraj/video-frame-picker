# Manual Test Cases

## Functional tests
1. Upload a valid MP4 video.
Expected: status shows extraction progress; 5 frames appear.

2. Click each of the 5 frames.
Expected: each click downloads one `.png` file.

3. Upload a very short video (less than 2 seconds).
Expected: 5 frames still appear; app does not crash.

4. Upload a second video after first one.
Expected: old frames are replaced with new frames.

5. Upload unsupported or corrupted video file.
Expected: user sees readable error message; app stays usable.

## UX tests
1. Verify on desktop browser (Chrome, Edge, Firefox).
Expected: layout is readable, buttons clickable.

2. Verify on mobile browser.
Expected: one-column responsive frame grid; no horizontal scroll.

## SEO tests
1. Open page source and confirm:
- `<title>` exists and is meaningful.
- `<meta name="description">` exists.
- Canonical URL points to your real domain.

2. Visit `/robots.txt`.
Expected: file loads and contains sitemap URL.

3. Visit `/sitemap.xml`.
Expected: valid XML with correct domain URL.

4. Run Google PageSpeed Insights.
Expected: page is crawlable and mobile-friendly.
