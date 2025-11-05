# Safety-AI
It's a safe and free , security AI.







bot.py:   (file name)

import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np


load_dotenv()


DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')


print("YOLO modeli yükleniyor...")
model = YOLO('yolov8n.pt')  # YOLOv8 nano model (hızlı ve hafif)
print("YOLO modeli yüklendi!")


intents = discord.Intents.default()
intents.message_content = True
intents.messages = True


bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} olarak giriş yapıldı!')
    print(f'Bot ID: {bot.user.id}')
    print('Bot hazır ve fotoğraf bekliyor...')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                try:
                    print(f"📸 Resim tespit edildi: {attachment.filename}")
                    async with message.channel.typing():
                        response_text, annotated_image = analyze_image_with_yolo(attachment.url)
                        print(f"✅ Analiz tamamlandı")
                        if annotated_image:
                            await message.reply(response_text, file=discord.File(annotated_image, filename='detected.jpg'))
                        else:
                            await message.reply(response_text)
                except Exception as e:
                    print(f"❌ Hata: {str(e)}")
                    await message.reply(f"❌ Hata oluştu: {str(e)}")
    await bot.process_commands(message)

def analyze_image_with_yolo(image_url):
    """YOLO kullanarak resmi analiz et ve nesneleri tespit et"""
    try:
        print(f"🔍 Resim analiz ediliyor: {image_url}")
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        img_array = np.array(image)
        results = model(img_array)
        detections = results[0]
        detected_objects = {}
        for box in detections.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            if confidence > 0.5:
                if class_name in detected_objects:
                    detected_objects[class_name]['count'] += 1
                    detected_objects[class_name]['confidences'].append(confidence)
                else:
                    detected_objects[class_name] = {
                        'count': 1,
                        'confidences': [confidence]
                    }
        turkish_names = {
            'person': 'kişi/insan',
            'bicycle': 'bisiklet',
            'car': 'araba',
            'motorcycle': 'motosiklet',
            'airplane': 'uçak',
            'bus': 'otobüs',
            'train': 'tren',
            'truck': 'kamyon',
            'boat': 'tekne',
            'traffic light': 'trafik ışığı',
            'fire hydrant': 'yangın musluğu',
            'stop sign': 'dur işareti',
            'parking meter': 'parkmetre',
            'bench': 'bank',
            'bird': 'kuş',
            'cat': 'kedi',
            'dog': 'köpek',
            'horse': 'at',
            'sheep': 'koyun',
            'cow': 'inek',
            'elephant': 'fil',
            'bear': 'ayı',
            'zebra': 'zebra',
            'giraffe': 'zürafa',
            'backpack': 'sırt çantası',
            'umbrella': 'şemsiye',
            'handbag': 'el çantası',
            'tie': 'kravat',
            'suitcase': 'valiz',
            'frisbee': 'frizbi',
            'skis': 'kayak',
            'snowboard': 'snowboard',
            'sports ball': 'spor topu',
            'kite': 'uçurtma',
            'baseball bat': 'beyzbol sopası',
            'baseball glove': 'beyzbol eldiveni',
            'skateboard': 'kaykay',
            'surfboard': 'sörf tahtası',
            'tennis racket': 'tenis raketi',
            'bottle': 'şişe',
            'wine glass': 'şarap kadehi',
            'cup': 'bardak/fincan',
            'fork': 'çatal',
            'knife': 'bıçak',
            'spoon': 'kaşık',
            'bowl': 'kase',
            'banana': 'muz',
            'apple': 'elma',
            'sandwich': 'sandviç',
            'orange': 'portakal',
            'broccoli': 'brokoli',
            'carrot': 'havuç',
            'hot dog': 'sosisli',
            'pizza': 'pizza',
            'donut': 'donut',
            'cake': 'pasta',
            'chair': 'sandalye',
            'couch': 'koltuk',
            'potted plant': 'saksı bitkisi',
            'bed': 'yatak',
            'dining table': 'yemek masası',
            'toilet': 'tuvalet',
            'tv': 'televizyon',
            'laptop': 'laptop',
            'mouse': 'fare (bilgisayar)',
            'remote': 'kumanda',
            'keyboard': 'klavye',
            'cell phone': 'cep telefonu',
            'microwave': 'mikrodalga',
            'oven': 'fırın',
            'toaster': 'ekmek kızartma makinesi',
            'sink': 'lavabo',
            'refrigerator': 'buzdolabı',
            'book': 'kitap',
            'clock': 'saat',
            'vase': 'vazo',
            'scissors': 'makas',
            'teddy bear': 'oyuncak ayı',
            'hair drier': 'saç kurutma makinesi',
            'toothbrush': 'diş fırçası'
        }
        if detected_objects:
            result_text = "� **YOLO Nesne Tespiti:**\n\n"
            result_text += f"**Toplam {len(detected_objects)} farklı nesne türü tespit edildi:**\n\n"
            for obj_name, obj_data in sorted(detected_objects.items(), key=lambda x: x[1]['count'], reverse=True):
                turkish_name = turkish_names.get(obj_name, obj_name)
                count = obj_data['count']
                avg_conf = sum(obj_data['confidences']) / len(obj_data['confidences']) * 100
                result_text += f"• **{turkish_name}** ({obj_name}): {count} adet (güven: %{avg_conf:.1f})\n"
            annotated_img = results[0].plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_img_rgb)
            buffer = BytesIO()
            annotated_pil.save(buffer, format='JPEG')
            buffer.seek(0)
            return result_text, buffer
        else:
            return "❌ Resimde herhangi bir nesne tespit edilemedi. Farklı bir resim deneyin.", None
    except Exception as e:
        return f"❌ Görsel analizi sırasında hata: {str(e)}", None

@bot.command(name='test')
async def test(ctx):
    """Bot'un çalıştığını test et"""
    await ctx.send('✅ Bot çalışıyor! Bir fotoğraf gönderin, analiz edeyim.')

@bot.command(name='yardim')
async def yardim(ctx):
    """Yardım mesajı göster"""
    help_text = """
    📸 **YOLO Nesne Tespit Botu**
    **Kullanım:**
    • Herhangi bir kanala fotoğraf gönderin
    • Bot YOLOv8 kullanarak nesneleri tespit edip işaretleyecek
    **Komutlar:**
    • `!test` - Bot'un çalışıp çalışmadığını kontrol et
    • `!yardim` - Bu yardım mesajını göster
    **Özellikler:**
    • 80+ farklı nesne türünü tanır
    • Tespit edilen nesneleri işaretli resim olarak gönderir
    • Türkçe nesne isimleri
    **Desteklenen formatlar:** PNG, JPG, JPEG, GIF, WEBP
    """
    await ctx.send(help_text)

# Bot'u çalıştır
if __name__ == '__main__':
    if not DISCORD_TOKEN:
        print("❌ HATA: DISCORD_TOKEN bulunamadı! .env dosyasını kontrol edin.")
    else:
        print("🚀 Bot başlatılıyor...")
        bot.run(DISCORD_TOKEN)



requirements.txt: (file name)


discord.py==2.3.2
python-dotenv==1.0.0
requests==2.31.0
Pillow==10.2.0
ultralytics==8.0.196
opencv-python==4.8.1.78
torch==2.1.0
torchvision==0.16.0


.gitignore:  (file name)

__pycache__/
*.py[cod]
*$py.class
*.so
.env
venv/
env/
*.log
.DS_Store

.env :     (file name)


DISCORD_TOKEN("Your Discord TOKEN")


FİNİSH(<--Do not write finish )

















