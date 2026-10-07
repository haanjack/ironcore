import json
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path('/home/hanjack/ironcore/ironcore/docs/notes')
output=root/'artifacts/2026-10-07-remediation'
errors=[];requests=[];checks=[]
with sync_playwright() as engine:
 browser=engine.chromium.launch(headless=True,args=['--disable-gpu','--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1100})
 page.on('pageerror',lambda error:errors.append(str(error)))
 page.on('request',lambda request:requests.append(request.url) if request.url.startswith(('http:','https:')) else None)
 page.goto((root/'profiling-remediation/profile_report.html').as_uri());page.wait_for_selector('#comparison tbody tr')
 assert page.locator('#run option').count()==4
 for run in range(4):
  page.select_option('#run',str(run))
  assert page.locator('#rank option').count()==2
  for rank in range(2):
   page.select_option('#rank',str(rank))
   assert page.locator('#cards').inner_text().strip()
   assert page.locator('#kernels tbody tr').count()>0
   checks.append(f'run{run}/rank{rank} table/timeline')
 page.select_option('#window','20');page.locator('#offset').fill('750')
 page.locator('#offset').dispatch_event('input');page.select_option('#phase','1')
 page.check('#allRanks');page.fill('#kernelSearch','nccl')
 assert page.locator('#kernels').inner_text().lower().find('nccl')>=0
 checks.append('filters/search/zoom/offset')
 page.click('#reset');page.fill('#kernelSearch','')
 page.screenshot(path=str(output/'profiler-desktop.png'),full_page=True)
 for viewport in [{'width':390,'height':844},{'width':768,'height':1024}]:
  page.set_viewport_size(viewport);page.wait_for_timeout(100)
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  checks.append(f'responsive {viewport["width"]}')
 page.screenshot(path=str(output/'profiler-mobile.png'),full_page=True)
 browser.close()
assert not errors and not requests,(errors,requests)
(output/'browser_check.json').write_text(json.dumps({'status':'passed','checks':checks,'javascript_errors':errors,'external_requests':requests},indent=2)+'\n')
print('Browser checks passed',len(checks))
