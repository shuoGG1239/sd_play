import webuiapi
from datetime import datetime

api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)

prompt = """
<lora:mari_NoobXL:1.1>, mari_(idol)_(blue_archive), 1girl, black_dress,black_hat,animal ears, frilled dress, long hair, sleeveless, aqua eyes,
, black legwear,simple_background,border
"""

ret = api.txt2img(prompt=prompt,
                  negative_prompt="",
                  seed=-1,
                  styles=["sdxl_common"],
                  cfg_scale=7,
                  sampler_name='Euler',
                  steps=20,
                  width=768,
                  height=1344,
                  )

now_str = datetime.now().strftime('%Y%m%d%H%M%S')
ret.image.save('noob_%s.png' % now_str)
