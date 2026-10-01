"""Generated geometric fixtures solely for software validation."""
from pathlib import Path
from PIL import Image, ImageDraw
from .common import write_json


def generate(root,n=24):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for i in range(n):
        im=Image.new('RGB',(240,320),(230,220,190));d=ImageDraw.Draw(im)
        # Different layout per image; same palette deliberately. Not saree identity ground truth.
        step=15+i*2
        for x in range(8,230,step):
            for y in range(8,300,step):
                if i%3==0:d.ellipse((x,y,x+6+i%9,y+10),(20,50,90))
                elif i%3==1:d.rectangle((x,y,x+5,y+12+i%7),(20,50,90))
                else:d.polygon([(x,y),(x+10,y+3),(x+3,y+12)],fill=(20,50,90))
        d.rectangle((2,2,237,317),outline=(90,30,20),width=3)
        im.save(root/f'fixture_{i:03d}.png')
    write_json(root/'SOURCE.json',{'source':'programmatically generated geometric software fixtures','real_dataset':False})
