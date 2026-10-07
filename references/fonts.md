# Шрифты по жанру — свои кандидаты для `--with`

Сначала — образцы установленной коллекции пользователя (`TS fonts --specimen --text "..."`): там жанровые
фансаб-шрифты, которых нет в Google (готика Moyenage / Dark Chancery, кисть Kashima, ...). Эта таблица — когда
нужного жанра в коллекции нет или хочется сравнить. Выбранные передай `--with "A,B,C"` (вес можно указать:
`"Old Standard TT:700"`, без него берётся ближайший к оригиналу): скрипт поставит их на лист рядом с текущим
шрифтом элемента и пометит `! thin` / `! full-width` / `! missing letters`, но не выкинет.

Все семейства ниже — Google Fonts с кириллицей (OFL, можно вшивать в релиз), скачиваются сами. В Google нет
настоящей готики с кириллицей.

| Жанр оригинала | Где встречается | Кандидаты |
|---|---|---|
| Готика, старина, фэнтези | названия серий, древние книги, свитки | (установленная готика) · Ponomar, Monomakh, Triodion, Pochaevsk (церковный устав) · Ruslan Display (вязь, очень жирный) · Old Standard TT · Kurale |
| Изящная антиква, тонкое минтё | имена на карточках, титры, письма | Cormorant Garamond / SC / Unicase · Playfair Display · Prata · Oranienbaum · Forum · Spectral · EB Garamond |
| Жирная дидона | заголовки, вывески «под старину» | Yeseva One · Playfair Display 800–900 · Playfair Display SC 900 |
| Книжная антиква | таблички, документы, книги | PT Serif · Lora · Literata · Merriweather · Source Serif 4 · Alice · Ledger · Brygada 1918 · Piazzolla; капитель — Vollkorn SC, Alegreya SC |
| Брусковые засечки | техника, вывески, табло | Roboto Slab · Bitter · Podkova · Kelly Slab |
| Обычный гротеск | экраны, интерфейсы, указатели | Montserrat · Golos Text · Rubik · Ubuntu · Arsenal · Geologica · Exo 2 |
| Узкий гротеск | плотные строки, газеты, постеры | Oswald · Roboto Condensed · Fira Sans Condensed · Yanone Kaffeesatz · Alumni Sans · Sofia Sans Extra Condensed |
| Округлый | милое, поп, детское | Comfortaa · Nunito · M PLUS Rounded 1c · Zen Maru Gothic |
| Жирный акцидентный | громкие титры, комедия, спорт | Russo One · Days One · Seymour One · Rubik Mono One · Unbounded 800–900 · Dela Gothic One |
| Рукопись, маркер | записки, доска, дневник | Marck Script · Bad Script · Caveat · Neucha · Pangolin · Amatic SC · Shantell Sans |
| Кисть, каллиграфия | надписи кистью, вывески | Comforter Brush · Lobster · Pacifico · Great Vibes · Zen Kurenaido (тонкая кисть) · Zen Antique (старое минтё) |
| Грязь, кровь, хоррор | граффити, проклятия, разруха | Rubik Dirt · Rubik Wet Paint · Rubik Distressed · Rubik Burned · Rubik Spray Paint · Rubik Beastly · Underdog |
| Пиксель, техно | игры, экраны, роботы | Pixelify Sans · Tiny5 · Press Start 2P (широкий, `! full-width` — норма) · Handjet · Jura · Tektur · Advent Pro · Rubik Glitch |
| Мультяшный | звуки, взрывы, пузыри | Rubik Bubbles · Rubik Puddles · Kablammo · Climate Crisis · Rubik Doodle Shadow |
| Машинка, моно | терминалы, досье | IBM Plex Mono · JetBrains Mono · Victor Mono |

Не брать: японские Google-шрифты, у которых кириллица моноширинная, в ширину иероглифа («Ц а р с т в о»), —
Klee One, Yuji Syuku, Yuji Boku, Kaisei Decol, Kosugi Maru, Train One, Rampart One, Reggae One, Stick,
DotGothic16, Hachi Maru Pop, Kiwi Maru (проверено 7 октября 2026, advance ≈ 1 em).

Шпаргалка — отправная точка. Новый удачный шрифт для жанра (выбранный пользователем или прошедший сборку)
допиши сюда на шаге 6.

## fonts-online.ru — третий источник (жанровые фансаб-шрифты)

Тысячи кириллических шрифтов по жанрам, у каждого — лицензия и автор. **Скачивание — только через капчу:
скачивает пользователь** (капчу не обходить, файлы шрифтов из предпросмотра сайта не вытаскивать). Ты
листаешь раздел во встроенном браузере и присылаешь 3–5 ссылок; пользователь скачивает,
`TS fonts --import <zip|ttf|папка>` кладёт их в библиотеку (оригиналы не трогает), дальше — `--with`.

Раздел с кириллицей по жанру: `https://fonts-online.ru/cyrillic-fonts?categories=<id>&sort_by=title&page=N`
(N с 0, по 18 шрифтов; фильтр `alphabet` на страницах `/categories/...` пропускает латинские).

| id | Раздел | Для чего |
|---|---|---|
| 44 | Готические (~195) | названия, фэнтези — Moyenage15SHA (выбран для названий BC), Cattedrale, Deutsch Gothic, GothicCyr |
| 659 | Средневековые | фэнтези-антиква: Calligrapher (таблички BC), Beaumarchais (карточки BC), CC Spellcaster |
| 654 | Римские и греческие | капитель, антиква: FoglihtenNo06 (OFL), Constantine |
| 65 | Старославянские | устав, вязь |
| 662 / 664 | Каллиграфические / кисти | рукопись, кисть |
| 67 / 64 | Ужасы / Сказочные | хоррор, детское |

Как смотреть быстро: поле «Введите текст» — клик, выделить, напечатать русский текст надписи, **Enter**
(программная подстановка значения ломает предпросмотр); текст запоминается на следующих страницах. Сетка
вместо списка — вставить стиль `sample-list {display:grid; grid-template-columns:repeat(3,1fr)}
typeface-teaser .typeface-teaser-info-options {display:none}`, окно 1600×1100, прокрутить на щелчок
(шрифты подгружаются при появлении в окне), подождать 3 с, снимок: 18 шрифтов за раз. `zoom` у страницы
не ставить — предпросмотр не грузится.

Лицензия на странице шрифта: «SIL OFL» / «бесплатный» — можно вшивать в релиз; «для персонального
использования», «All rights reserved», без лицензии, «Demo» (часто без части букв) — скажи пользователю,
решает он.
