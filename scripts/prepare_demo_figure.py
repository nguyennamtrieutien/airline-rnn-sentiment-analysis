"""Arrange two crops of the real browser capture for readable document layout."""
from pathlib import Path
from PIL import Image, ImageDraw
root=Path(__file__).resolve().parents[1]
source=Image.open(root/'results/figures/F23_demo.png').convert('RGB')
# Coordinates apply to the verified 1000 x 2482 browser capture.
if source.size != (1000,2482):
    raise ValueError('Capture the documented desktop viewport again before arranging this figure.')
left=source.crop((30,628,970,1472))
right=source.crop((30,1484,970,2408))
canvas=Image.new('RGB',(1920,1100),'white')
canvas.paste(left,(0,36));canvas.paste(right,(980,36))
d=ImageDraw.Draw(canvas)
d.text((10,10),'(a) Input, prediction and inference',fill='#16344c')
d.text((990,10),'(b) Eight actual analysis stages',fill='#16344c')
canvas.save(root/'results/figures/F23_demo_report.png')
canvas.save(root/'results/figures/F23_demo_report.pdf',resolution=150)
source.save(root/'results/figures/F23_demo.pdf',resolution=150)
print('Prepared report crop from the real browser capture; no prediction values changed.')
