from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup
from mpi4py import MPI


BASE_URL = "https://en.wikipedia.org"
PERIODIC_TABLE_URL = f"{BASE_URL}/wiki/Periodic_table"
USER_AGENT = "MP-Lab3-MPI/1.0 (educational web parsing)"

ELEMENTS = [
    ("H", "Hydrogen"), ("He", "Helium"), ("Li", "Lithium"), ("Be", "Beryllium"),
    ("B", "Boron"), ("C", "Carbon"), ("N", "Nitrogen"), ("O", "Oxygen"),
    ("F", "Fluorine"), ("Ne", "Neon"), ("Na", "Sodium"), ("Mg", "Magnesium"),
    ("Al", "Aluminium"), ("Si", "Silicon"), ("P", "Phosphorus"), ("S", "Sulfur"),
    ("Cl", "Chlorine"), ("Ar", "Argon"), ("K", "Potassium"), ("Ca", "Calcium"),
    ("Sc", "Scandium"), ("Ti", "Titanium"), ("V", "Vanadium"), ("Cr", "Chromium"),
    ("Mn", "Manganese"), ("Fe", "Iron"), ("Co", "Cobalt"), ("Ni", "Nickel"),
    ("Cu", "Copper"), ("Zn", "Zinc"), ("Ga", "Gallium"), ("Ge", "Germanium"),
    ("As", "Arsenic"), ("Se", "Selenium"), ("Br", "Bromine"), ("Kr", "Krypton"),
    ("Rb", "Rubidium"), ("Sr", "Strontium"), ("Y", "Yttrium"), ("Zr", "Zirconium"),
    ("Nb", "Niobium"), ("Mo", "Molybdenum"), ("Tc", "Technetium"), ("Ru", "Ruthenium"),
    ("Rh", "Rhodium"), ("Pd", "Palladium"), ("Ag", "Silver"), ("Cd", "Cadmium"),
    ("In", "Indium"), ("Sn", "Tin"), ("Sb", "Antimony"), ("Te", "Tellurium"),
    ("I", "Iodine"), ("Xe", "Xenon"), ("Cs", "Caesium"), ("Ba", "Barium"),
    ("La", "Lanthanum"), ("Ce", "Cerium"), ("Pr", "Praseodymium"), ("Nd", "Neodymium"),
    ("Pm", "Promethium"), ("Sm", "Samarium"), ("Eu", "Europium"), ("Gd", "Gadolinium"),
    ("Tb", "Terbium"), ("Dy", "Dysprosium"), ("Ho", "Holmium"), ("Er", "Erbium"),
    ("Tm", "Thulium"), ("Yb", "Ytterbium"), ("Lu", "Lutetium"), ("Hf", "Hafnium"),
    ("Ta", "Tantalum"), ("W", "Tungsten"), ("Re", "Rhenium"), ("Os", "Osmium"),
    ("Ir", "Iridium"), ("Pt", "Platinum"), ("Au", "Gold"), ("Hg", "Mercury"),
    ("Tl", "Thallium"), ("Pb", "Lead"), ("Bi", "Bismuth"), ("Po", "Polonium"),
    ("At", "Astatine"), ("Rn", "Radon"), ("Fr", "Francium"), ("Ra", "Radium"),
    ("Ac", "Actinium"), ("Th", "Thorium"), ("Pa", "Protactinium"), ("U", "Uranium"),
    ("Np", "Neptunium"), ("Pu", "Plutonium"), ("Am", "Americium"), ("Cm", "Curium"),
    ("Bk", "Berkelium"), ("Cf", "Californium"), ("Es", "Einsteinium"), ("Fm", "Fermium"),
    ("Md", "Mendelevium"), ("No", "Nobelium"), ("Lr", "Lawrencium"), ("Rf", "Rutherfordium"),
    ("Db", "Dubnium"), ("Sg", "Seaborgium"), ("Bh", "Bohrium"), ("Hs", "Hassium"),
    ("Mt", "Meitnerium"), ("Ds", "Darmstadtium"), ("Rg", "Roentgenium"),
    ("Cn", "Copernicium"), ("Nh", "Nihonium"), ("Fl", "Flerovium"), ("Mc", "Moscovium"),
    ("Lv", "Livermorium"), ("Ts", "Tennessine"), ("Og", "Oganesson"),
]


def fetch(url: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html"
    cache_path = cache_dir / key
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    text = response.text
    cache_path.write_text(text, encoding="utf-8")
    return text


def extract_element_links(periodic_html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(periodic_html, "lxml")
    wanted = {
        name.lower(): {"symbol": symbol, "name": name}
        for symbol, name in ELEMENTS
        if len(symbol) == 2
    }
    links: dict[str, dict[str, str]] = {}

    for anchor in soup.select("a[href^='/wiki/']"):
        label = anchor.get_text(" ", strip=True)
        element = wanted.get(label.lower())
        if not element:
            continue
        symbol = element["symbol"]
        href = anchor.get("href", "")
        if ":" in href or "#" in href:
            continue
        links[symbol] = {
            "symbol": symbol,
            "name": element["name"],
            "url": urljoin(BASE_URL, href),
        }

    return sorted(links.values(), key=lambda item: item["symbol"])


def main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    content = soup.find(id="mw-content-text") or soup.find("main") or soup

    for selector in [
        "style", "script", "noscript", "table.infobox", "table.sidebar", "table.navbox",
        ".navbox", ".sidebar", ".metadata", ".mw-editsection", ".reference",
        ".reflist", ".hatnote", ".ambox", ".sistersitebox",
    ]:
        for node in content.select(selector):
            node.decompose()

    stop_ids = {"See_also", "References", "Further_reading", "External_links", "Notes"}
    for heading in list(content.find_all(["h2", "h3"])):
        headline = heading.find(class_="mw-headline")
        heading_id = headline.get("id") if headline else heading.get("id")
        if heading_id in stop_ids:
            for node in list(heading.find_all_next()):
                node.decompose()
            heading.decompose()
            break

    return content.get_text(" ", strip=True)


def count_symbol(text: str, symbol: str) -> int:
    return len(re.findall(rf"(?<![A-Za-z]){re.escape(symbol)}(?![a-z])", text))


def process_page(item: dict[str, str], symbols: list[str], cache_dir: Path) -> dict:
    html = fetch(item["url"], cache_dir)
    text = main_text(html)
    counts = {symbol: count_symbol(text, symbol) for symbol in symbols}
    return {
        "symbol": item["symbol"],
        "name": item["name"],
        "url": item["url"],
        "own_count": counts[item["symbol"]],
        "all_symbol_counts": counts,
        "text_chars": len(text),
    }


def chunks(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[i::size] for i in range(size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="lab3_results/run.json")
    parser.add_argument("--cache-dir", default=".cache/wiki")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    cache_dir = Path(args.cache_dir)
    start = time.perf_counter()

    if rank == 0:
        periodic_html = fetch(PERIODIC_TABLE_URL, cache_dir)
        elements = extract_element_links(periodic_html)
        work = chunks(elements, size)
        symbols = [item["symbol"] for item in elements]
    else:
        elements = None
        work = None
        symbols = None

    symbols = comm.bcast(symbols, root=0)
    assigned = comm.scatter(work, root=0)
    local_results = [process_page(item, symbols, cache_dir) for item in assigned]
    gathered = comm.gather(local_results, root=0)

    if rank != 0:
        return

    page_results = [item for part in gathered for item in part]
    total_counter: Counter[str] = Counter()
    for result in page_results:
        total_counter.update(result["all_symbol_counts"])

    elapsed = time.perf_counter() - start
    own_top = sorted(
        page_results,
        key=lambda row: (-row["own_count"], row["symbol"]),
    )[:5]
    global_top = [
        {"symbol": symbol, "count": count}
        for symbol, count in total_counter.most_common(5)
    ]

    payload = {
        "processes": size,
        "elapsed_s": elapsed,
        "periodic_table_url": PERIODIC_TABLE_URL,
        "elements_processed": len(page_results),
        "top_own_page_mentions": [
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "count": row["own_count"],
                "url": row["url"],
            }
            for row in own_top
        ],
        "top_global_symbol_mentions": global_top,
        "pages": sorted(page_results, key=lambda row: row["symbol"]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"processes={size}")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"elements_processed={len(page_results)}")
    print("top_own_page_mentions=" + json.dumps(payload["top_own_page_mentions"], ensure_ascii=False))
    print("top_global_symbol_mentions=" + json.dumps(global_top, ensure_ascii=False))


if __name__ == "__main__":
    main()
