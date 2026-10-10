# ฟอนต์ไทยที่มากับโปรแกรม

**47 ตระกูล, 358 แบบน้ำหนัก/ตัวเอียง** ใช้ได้ออฟไลน์ ไม่ต้องติดตั้งลง Windows

- 34 ตระกูลจาก Google Fonts (คลังฟอนต์ของ Kotoba Studio เพิ่มเมื่อ 8 ตุลาคม 2026)
- 13 ตระกูลจากชุด TLWG (Thai Linux Working Group) รุ่น 0.7.4 เพิ่มเมื่อ 10 ตุลาคม 2026

## วิธีเลือกในโปรแกรม

เปิดรายการ **แบบอักษร/ฟอนต์** แล้วค้นหาชื่อ เช่น `Itim`, `Sarabun`, `Purisa` หรือชื่อไทย เช่น `ไอติม` รายการนี้สแกนโฟลเดอร์ฟอนต์ใหม่ทุกครั้งที่เปิด ไม่ต้องรีสตาร์ตโปรแกรมเพื่อค้นพบไฟล์ใหม่

รายการแสดงชื่อตระกูลตรง ๆ เช่น `Itim` หรือ `Sarabun - Bold` เลือกน้ำหนักและตัวเอียงจากรายการของตระกูลนั้น

| งานมังงะ              | ฟอนต์ที่ควรเริ่มลอง                     | หมายเหตุ                                   |
| --------------------- | --------------------------------------- | ------------------------------------------ |
| บทพูดทั่วไป           | Itim, Mali, Sawasdee                    | บุคลิกคล้ายลายมือ เหมาะกับบอลลูนคำพูด      |
| บทพูดที่ต้องอ่านชัด   | Sarabun, Noto Sans Thai Looped, Garuda  | มีหัวอักษรชัด เหมาะกับตัวหนังสือเล็ก       |
| บทบรรยาย              | Pridi, Noto Serif Thai, Kinnari, Norasi | มีบุคลิกเหมาะกับข้อความเล่าเรื่อง          |
| ตะโกน / เสียงเอฟเฟกต์ | Kanit, Mitr, Chonburi                   | เลือก Bold หรือ Black เมื่อมีแบบนั้น       |
| ความคิด / ลายมือ      | Mali, Sriracha, Purisa                  | ลองเพิ่มระยะบรรทัดเมื่อมีวรรณยุกต์หลายชั้น |

## ฟอนต์ทั้งหมด

**จาก Google Fonts:** Anuphan, Athiti, Bai Jamjuree, Chakra Petch, Charm, Charmonman, Chonburi, Fahkwang, Google Sans, IBM Plex Sans Thai, IBM Plex Sans Thai Looped, Itim, K2D, Kanit, Kodchasan, KoHo, Krub, Maitree, Mali, Mitr, Niramit, Noto Sans Thai, Noto Sans Thai Looped, Noto Serif Thai, Pattaya, Playpen Sans Thai, Pridi, Prompt, Sarabun, Sriracha, Srisakdi, Taviraj, Thasadith, Trirong

**จากชุด TLWG:** Garuda, Kinnari, Laksaman, Loma, Norasi, Purisa, Sawasdee, Tlwg Mono, Tlwg Typewriter, Tlwg Typist, Tlwg Typo, Umpush, Waree

## เพิ่มฟอนต์ของคุณเอง

วางไฟล์ `.ttf` หรือ `.otf` ในโฟลเดอร์ `fonts` นี้ แล้วเปิดรายการฟอนต์ใหม่ ฟอนต์ที่ติดตั้งใน Windows ก็เลือกได้เช่นกัน

ฟอนต์ที่ใบอนุญาตไม่อนุญาตให้แจกจ่ายต่อ (เช่น ฟอนต์ที่ใช้ฟรีเฉพาะงานส่วนตัว) ไม่ได้รวมมากับโปรแกรม ให้ดาวน์โหลดจากเว็บของผู้ออกแบบและยอมรับสัญญาอนุญาตด้วยตัวเอง ไฟล์ที่วางเพิ่มในโฟลเดอร์นี้ไม่ถูกส่งขึ้น repo ของโปรเจกต์

## แหล่งที่มาและใบอนุญาต

**ชุด Google Fonts** ต้นทางเป็น WOFF2 แยกชุดอักษร Thai, Latin และ Latin-ext จึงรวมชุดอักษรและแปลงเป็น TTF ด้วย fontTools ส่วน Noto Sans Thai และ Noto Serif Thai สร้างเป็นน้ำหนักคงที่จาก variable font ทุกตระกูลใช้ใบอนุญาต SIL OFL 1.1 สำเนาอยู่ใน [licenses](licenses) ไฟล์ที่แปลงใช้ชื่อตระกูลเดิม ไม่มีตระกูลใดสงวนชื่อฟอนต์ของตัวเองไว้ในใบอนุญาต (Pattaya สงวนชื่อ Lobster ซึ่งไม่ได้ใช้)

รุ่นก่อนหน้าตั้งชื่อตระกูลขึ้นต้นด้วย `MangaX` การตั้งค่าและโปรเจกต์ที่บันทึกชื่อแบบนั้นไว้ยังเปิดได้ตามปกติ โปรแกรมจับคู่กับชื่อใหม่ให้เอง

**ชุด TLWG** เป็นไฟล์ต้นฉบับไม่ดัดแปลงจาก [fonts-tlwg v0.7.4](https://github.com/tlwg/fonts-tlwg/releases/tag/v0.7.4) (`ttf-tlwg-0.7.4.zip`) ใช้ใบอนุญาต GPL รุ่น 2 ขึ้นไปพร้อมข้อยกเว้นสำหรับการฝังฟอนต์ในเอกสาร ยกเว้น Waree ที่ใช้ใบอนุญาตของ Bitstream Vera ข้อความเต็มอยู่ที่ [licenses/tlwg-COPYING.txt](licenses/tlwg-COPYING.txt)

รายการไฟล์และค่า SHA-256 ของทุกแบบอยู่ใน [thai-font-catalog.json](thai-font-catalog.json)
