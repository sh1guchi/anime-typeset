# Шрифты по жанру — свои кандидаты для `--with`

Перед `TS fonts --match` назови 2–4 шрифта под жанр надписи и передай их `--with "A,B,C"` (вес можно
указать: `"Old Standard TT:700"`, без него берётся ближайший к оригиналу). Скрипт замерит их как свои
кандидаты, поставит на лист сразу после текущего шрифта элемента и пометит, что не прошло бы его фильтры
(`! thin`, `! full-width`, `! missing letters`), но не выкинет.

Все семейства ниже — Google Fonts с кириллицей (OFL, можно вшивать в релиз), скачиваются сами. Из
установленных у пользователя смотри `TS fonts --cyr --grep <часть имени>`: там бывают свои находки
(например, готический Moyenage в названиях Black Clover — в Google настоящей готики с кириллицей нет).

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
