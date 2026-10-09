<div align="center">

# MangaX

**โปรแกรมแปลมังงะบนเดสก์ท็อป ปรับปรุงสำหรับคอมมูนิตี้นักแปลไทย**

ตรวจจับข้อความ → OCR → แปล → ลบข้อความต้นฉบับ → จัดตัวอักษร พร้อมตัวแก้ไขในตัว

[![Tests](https://github.com/nousx/MangaX/actions/workflows/tests.yml/badge.svg)](https://github.com/nousx/MangaX/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-GPL--3.0-red)](LICENSE.txt)
[![Based on](https://img.shields.io/badge/based%20on-manga--translator--ui-green)](https://github.com/hgmzhn/manga-translator-ui)

**ภาษา / Language**: ไทย | [English](README_EN.md) | [简体中文](README_ZH.md)

</div>

## MangaX คืออะไร

MangaX เป็น fork ของ [Manga Translator UI](https://github.com/hgmzhn/manga-translator-ui) โดย **hgmzhn** ซึ่งต่อยอดจาก [manga-image-translator](https://github.com/zyddnys/manga-image-translator) โดย **zyddnys** งานหลักทั้งหมด ทั้งระบบตรวจจับ OCR ซ่อมภาพ เรนเดอร์ และตัวแก้ไข เป็นผลงานของโครงการต้นทาง MangaX เพิ่มส่วนที่คนแปลมังงะภาษาไทยต้องใช้

`README_EN.md` และ `README_ZH.md` เป็นเอกสารของโครงการต้นทาง ลิงก์ติดตั้งและ Docker ในนั้นชี้ไปที่ต้นทาง ไม่ใช่ MangaX

## สิ่งที่ MangaX เพิ่ม

- **เมนูภาษาไทย** ครบทั้งโปรแกรม ใช้คำของงานแปลมังงะ
- **ฟอนต์ไทย 34 ตระกูล 300 แบบ** ติดมากับโปรแกรม ใช้ออฟไลน์ได้ (สัญญาอนุญาต OFL อยู่ใน `fonts/licenses/`)
- **ตัวเลือกฟอนต์แสดงข้อความจริง** แต่ละแถวแสดงบทพูดของกรอบที่เลือกด้วยฟอนต์นั้น แบ่งหมวด บทพูด / ลายมือ / บรรยาย / เอฟเฟกต์ และมีรายการโปรด
- **แปลผ่าน Codex CLI** ด้วยบัญชี ChatGPT ที่ล็อกอินไว้ ไม่ต้องกรอก API key
- **OCR และซ่อมภาพในเครื่อง** ทำงานได้โดยไม่ต้องเชื่อมบริการแปล
- **แท็บผู้ช่วย** ในตัวแก้ไข: OCR ทั้งหน้า, คิวงานทั้งบทที่ทำต่อจากจุดค้างได้, บริบทเรื่องและเสียงตัวละครสำหรับตัวแปล, ตรวจคุณภาพก่อนส่งออก
- **ปรับความปลอดภัยของโหมดเว็บ** เช่น host เริ่มต้นเป็น `127.0.0.1`

## ติดตั้ง (Windows)

ดาวน์โหลด [**MangaX-Setup.exe**](https://github.com/nousx/MangaX/releases/latest/download/MangaX-Setup.exe) แล้วเปิด ตัวติดตั้งจะตรวจการ์ดจอของเครื่อง เลือกชุดที่ตรงกัน (NVIDIA CUDA 13.0 / CUDA 12.6, AMD ROCm หรือ CPU) ดาวน์โหลด แตกไฟล์ และสร้างทางลัดให้ ถ้าเน็ตหลุดกลางทาง เปิดใหม่แล้วกดติดตั้งอีกครั้งจะโหลดต่อจากจุดเดิม

ไฟล์ยังไม่ได้เซ็นลายเซ็นดิจิทัล Windows SmartScreen จะเตือนตอนเปิดครั้งแรก กด **More info** แล้ว **Run anyway**

ต้องการโหลดเองทีละชุดก็ได้จากหน้า [Releases](https://github.com/nousx/MangaX/releases) ชุดที่มีหลายไฟล์ต้องโหลดให้ครบแล้วแตกจาก `.001`

## ติดตั้งจากซอร์ส (Windows)

ต้องมี [uv](https://docs.astral.sh/uv/) และ Git

```powershell
git clone https://github.com/nousx/MangaX.git
cd MangaX
uv python install 3.12
# การ์ดจอ NVIDIA: ใช้ --group cuda13.0 หรือ --group cuda12.6 แทน cpu
uv sync --locked --no-default-groups --group cpu
uv run --no-sync python desktop_qt_ui\main.py
```

โมเดลจะดาวน์โหลดครั้งแรกที่ใช้งาน ไฟล์โมเดลโฮสต์โดยโครงการต้นทาง

## เริ่มใช้งานแบบสั้น

1. เปิดภาพในหน้าแก้ไข
2. สร้างหรือเลือกกรอบข้อความ เลือกโมเดล `48px` แล้วกด **อ่านข้อความ**
3. เลือกบริการแปลและภาษาเป้าหมาย **ไทย** แล้วกด **แปล**
4. เลือกฟอนต์จากรายการที่แสดงบทพูดจริง ปรับตำแหน่ง แล้วส่งออก

การแปลด้วย Codex ต้องติดตั้ง Codex CLI และรัน `codex login` ก่อน การแปลใช้อินเทอร์เน็ตและโควตาของบัญชี ส่งเฉพาะข้อความ OCR ไม่ส่งภาพ

## ร่วมพัฒนา

เปิด [Issue](https://github.com/nousx/MangaX/issues) หรือส่ง Pull Request ได้ ทั้งคำแปลเมนู ฟอนต์ และฟีเจอร์สำหรับงานแปลไทย

รันเทสต์:

```powershell
uv sync --locked --no-default-groups --group cpu --group test
uv run --no-sync pytest test
```

## เครดิต

- [hgmzhn/manga-translator-ui](https://github.com/hgmzhn/manga-translator-ui) — โปรแกรมต้นทางที่ MangaX แยกออกมา หากโปรแกรมนี้มีประโยชน์ โปรดสนับสนุนผู้พัฒนาต้นทาง ช่องทางอยู่ในหน้า **เกี่ยวกับ** ของโปรแกรม
- [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) — เอนจินแปลภาพมังงะ
- โมเดลและไลบรารีที่โครงการต้นทางใช้ ดูรายการใน [README_EN.md](README_EN.md)
- ฟอนต์ไทยจาก Google Fonts และผู้ออกแบบแต่ละราย ภายใต้ SIL Open Font License

## สัญญาอนุญาต

[GPL-3.0](LICENSE.txt) เช่นเดียวกับโครงการต้นทาง ซอร์สโค้ดที่แก้ไขเผยแพร่ใน repository นี้
