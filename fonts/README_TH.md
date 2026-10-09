# ฟอนต์ไทยสำหรับ MangaX

เพิ่มจากคลังฟอนต์ของ Kotoba Studio ใน `C:\Users\SpecTruM\Desktop\Working\manga-system` เมื่อ 8 ตุลาคม 2026: **34 ตระกูล, 300 แบบน้ำหนัก/ตัวเอียง**, ขนาดไฟล์ฟอนต์รวมประมาณ 28 MiB

## วิธีเลือกในโปรแกรม

เปิดรายการ **แบบอักษร/ฟอนต์** อีกครั้ง แล้วค้นหา `MangaX` หรือชื่อ เช่น `Itim`, `Sarabun`, `Kanit` รายการนี้สแกนโฟลเดอร์ฟอนต์ใหม่เมื่อเปิด ไม่จำเป็นต้องติดตั้งฟอนต์ลง Windows และไม่ต้องรีสตาร์ตโปรแกรมเพื่อค้นพบไฟล์ใหม่

รายการจะแสดงชื่อเช่น `MangaX Itim` หรือ `MangaX Sarabun - Bold` เลือกน้ำหนักและตัวเอียงที่ต้องการจากรายการของตระกูลนั้น ใช้ได้ในพรีวิวตัวแก้ไขและภาพส่งออก และใช้ฟอนต์ได้ออฟไลน์

| งานมังงะ | ฟอนต์ที่ควรเริ่มลอง | หมายเหตุ |
|---|---|---|
| บทพูดทั่วไป | MangaX Itim, MangaX Mali | บุคลิกคล้ายลายมือ เหมาะกับบอลลูนคำพูด |
| บทพูดที่ต้องอ่านชัด | MangaX Sarabun, MangaX Noto Sans Thai Looped | มีหัวอักษรชัด เหมาะกับตัวหนังสือเล็ก |
| บทบรรยาย | MangaX Pridi, MangaX Noto Serif Thai | มีบุคลิกเหมาะกับข้อความเล่าเรื่อง |
| ตะโกน / เสียงเอฟเฟกต์ | MangaX Kanit, MangaX Mitr, MangaX Chonburi | เลือก Bold หรือ Black เมื่อมีแบบนั้น |
| ความคิด / น้ำเสียงนุ่ม | MangaX Mali, MangaX Sriracha | ลองระยะบรรทัดเพิ่มเมื่อมีวรรณยุกต์หลายชั้น |

ดูภาพเปรียบเทียบทุกตระกูลที่ [thai-font-preview.png](../../security-audit/thai-font-preview.png) ตัวอย่างสร้างจากระบบเรนเดอร์ภาพส่งออกของโปรแกรม ไม่ใช่ภาพจากเว็บ

## ฟอนต์ทั้งหมด

Anuphan, Athiti, Bai Jamjuree, Chakra Petch, Charm, Charmonman, Chonburi, Fahkwang, Google Sans, IBM Plex Sans Thai, IBM Plex Sans Thai Looped, Itim, K2D, Kanit, Kodchasan, KoHo, Krub, Maitree, Mali, Mitr, Niramit, Noto Sans Thai, Noto Sans Thai Looped, Noto Serif Thai, Pattaya, Playpen Sans Thai, Pridi, Prompt, Sarabun, Sriracha, Srisakdi, Taviraj, Thasadith, Trirong

## แหล่งที่มาและการแปลง

ต้นทางใช้ WOFF2 แยกชุดอักษร Thai, Latin และ Latin-ext โปรแกรม Qt ใช้ไฟล์ desktop จึงรวมชุดอักษรและแปลงเป็น TTF ด้วย fontTools ส่วน Noto Sans Thai และ Noto Serif Thai ถูกสร้างเป็นน้ำหนักคงที่จาก variable font ใช้วิธีตามเอกสาร [fontTools Merger](https://fonttools.readthedocs.io/en/latest/merge.html) และ [Variable font instancer](https://fonttools.readthedocs.io/en/latest/varLib/instancer.html)

สำเนาที่แปลงใช้ชื่อตระกูลขึ้นต้น `MangaX` เพื่อแยกจากต้นฉบับ เก็บข้อความลิขสิทธิ์เดิมในข้อมูลฟอนต์ พร้อมสำเนาใบอนุญาต OFL-1.1 ของแต่ละตระกูลใน [licenses](licenses) เก็บรายการไฟล์ต้นทางและ SHA-256 ของทุกไฟล์ใน [thai-font-catalog.json](thai-font-catalog.json)

ตัวแปลงอยู่ที่ [import_kotoba_fonts.py](../../security-audit/import_kotoba_fonts.py) และปฏิเสธการเขียนทับฟอนต์ที่มีอยู่แล้ว

## ผลตรวจ

ตรวจครบ 300 แบบผ่านตัวเลือกฟอนต์ Qt และระบบเรนเดอร์ภาพส่งออกของ MangaX: ชื่อตระกูล น้ำหนัก และตัวเอียงตรงกับไฟล์; ข้อความตัวอย่างภาษาไทยรวมสระ/วรรณยุกต์และอังกฤษไม่มี glyph หายหรือการสลับไปใช้ฟอนต์อื่น; มีภาพ RGBA ส่งออกจริงครบทุกแบบ

รายงาน [thai-font-verification.json](../../security-audit/thai-font-verification.json) และสคริปต์ [verify_imported_fonts.py](../../security-audit/verify_imported_fonts.py) การทดสอบนี้ครอบคลุมข้อความตัวอย่าง ไม่ใช่การรับรองทุกอักขระหรือทุกขนาดบอลลูน
