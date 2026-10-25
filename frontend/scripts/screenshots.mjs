// Visual smoke test: drives the mock app through several states and saves screenshots.
// Usage: npm run dev (in another shell) then `node scripts/screenshots.mjs [baseUrl] [outDir]`
import { chromium } from 'playwright'
import fs from 'node:fs'

const base = process.argv[2] ?? 'http://localhost:5173'
const out = process.argv[3] ?? new URL('./out/', import.meta.url).pathname
fs.mkdirSync(out, { recursive: true })

const browser = await chromium.launch()
const [vw, vh] = (process.env.VIEWPORT ?? '1600x900').split('x').map(Number)
const page = await browser.newPage({ viewport: { width: vw, height: vh }, deviceScaleFactor: 1 })
page.on('pageerror', (e) => console.error('PAGE ERROR', e.message))
page.on('console', (m) => m.type() === 'error' && console.error('console:', m.text()))

const shot = async (name) => {
  await page.screenshot({ path: `${out}/${name}.png` })
  console.log('saved', name)
}
const rig = (cards) => page.evaluate((c) => window.__mock.rig(c), cards)
const waitDecision = () => page.waitForSelector('.dock.actions', { timeout: 15000 })
const waitSettled = () => page.waitForSelector('.dock.betting .chip-rack', { timeout: 20000 }).then(() => page.waitForSelector('.result-chip', { timeout: 20000 }))
const bet = async (spots, chip = 10) => {
  await page.click(`button[aria-label="€${chip} chip"]`)
  for (const s of spots) await page.click(`.spot-${s}`)
}

await page.goto(`${base}/?mock=1`)
await page.waitForSelector('.stage')
await page.waitForTimeout(600)

// 1. betting with chips placed
await bet(['main', 'main', 'perfect_pairs', 'twenty_one_plus_three', 'bust_bonus'])
await page.click('button[aria-label="€5 chip"]')
await page.click('.spot-crazy_7')
await shot('01-betting')

// 2. split hand + crowd + advice
await page.click('text=Advice')
await rig(['8_of_hearts', '6_of_spades', '8_of_diamonds', '10_of_clubs', '3_of_clubs', '3_of_hearts', '9_of_spades', 'K_of_hearts', '7_of_clubs'])
await page.keyboard.press('Space')
await waitDecision()
await page.waitForTimeout(500)
await page.keyboard.press('p')
await page.waitForSelector('.hand.split.active', { timeout: 15000 })
await waitDecision()
await page.waitForTimeout(700)
await shot('02-split-midhand')

// 3. finish: double hand 1, stand hand 2 → dealer busts with 3 cards
await page.keyboard.press('d')
await page.waitForTimeout(1500)
await waitDecision()
await page.keyboard.press('s')
await waitSettled()
await page.waitForTimeout(900)
await shot('03-settled-sidebets')

// 4. crowd cards: stand on 12 vs 9, crowd keeps hitting
await rig(['10_of_spades', '9_of_hearts', '2_of_clubs', '8_of_diamonds', '4_of_hearts', '5_of_clubs', '10_of_hearts'])
await page.keyboard.press('Space')
await waitDecision()
await page.waitForTimeout(400)
await page.keyboard.press('s')
await waitSettled()
await page.waitForTimeout(900)
await shot('04-crowd-cards')

// 5. insurance prompt
await rig(['10_of_diamonds', 'A_of_spades', '9_of_clubs', '6_of_hearts'])
await page.keyboard.press('Space')
await page.waitForSelector('.dock-q', { timeout: 15000 })
await page.waitForTimeout(500)
await shot('05-insurance')
await page.keyboard.press('n')
await waitDecision()
await page.waitForTimeout(400)
await shot('06-player-advice-cashout')
await page.keyboard.press('s')
await waitSettled()

// 6. autoplay a bit then simulate
await page.click('.ai-controls .toggle:has-text("Autoplay")')
await page.waitForTimeout(9000)
await shot('07-autoplay-live-network')
await page.click('.ai-controls .toggle:has-text("Autoplay")')
await page.waitForTimeout(1500)
await page.click('.tabs button:has-text("Metrics")')
await page.click('.sim-row .seg button:has-text("100k")')
await page.click('.sim-row button:has-text("Run")')
await page.waitForTimeout(3000)
await shot('08-simulate-running')
await page.waitForFunction(() => document.querySelector('.sim-row .btn')?.textContent === 'Run', null, { timeout: 120000 })
await page.waitForTimeout(500)
await shot('09-simulate-done')
await page.click('.tabs button:has-text("Training")')
await page.waitForTimeout(800)
await shot('10-training')
await page.click('.pt-btn')
await page.waitForTimeout(300)
await shot('11-paytable')

// 7. scrolled panels (features / composition / analytics / history)
await page.keyboard.press('Escape')
await page.click('.pt-btn')
await page.click('.tabs button:has-text("Network")')
await page.click('.collapser')
await page.evaluate(() => document.querySelectorAll('.panel-scroll').forEach((el) => (el.scrollTop = el.scrollHeight)))
await page.waitForTimeout(400)
await shot('12-panels-scrolled')

// 8. fast autoplay until the cut card of half 1 → half 2 (count carries over)
await page.focus('input[aria-label="Autoplay speed"]')
await page.keyboard.press('End')
await page.click('.ai-controls .toggle:has-text("Autoplay")')
await page.waitForFunction(() => document.querySelector('.half.dealing .half-name')?.textContent?.includes('Half 2'), null, { timeout: 180000 })
await page.waitForTimeout(6000)
await page.click('.ai-controls .toggle:has-text("Autoplay")')
await page.waitForTimeout(1500)
await page.evaluate(() => document.querySelectorAll('.panel-scroll').forEach((el) => (el.scrollTop = 0)))
await shot('13-half-two')

await browser.close()
