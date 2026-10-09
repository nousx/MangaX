"""Bounded, explicit manga context shared by Codex and the chapter editor."""
import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    default=str).encode('utf-8')).hexdigest()


def context_path(image_path, root=None):
    from manga_translator.runtime_paths import get_application_dir
    folder = os.path.normcase(os.path.abspath(os.path.dirname(str(image_path))))
    return Path(root or get_application_dir()) / 'config' / 'chapter_context' / (fingerprint(folder) + '.json')


def load_chapter_context(image_path, root=None):
    if not image_path:
        return {}
    path = context_path(image_path, root)
    if not path.is_file():
        return {}
    if path.stat().st_size > 512_000:
        raise ValueError('Chapter context is too large.')
    value = json.loads(path.read_text(encoding='utf-8'))
    return value if isinstance(value, dict) else {}


def glossary_entries(prompt):
    """Accept the application's categorized HQ glossary and flat entries."""
    if not isinstance(prompt, dict):
        return []
    glossary = prompt.get('glossary', [])
    entries = [item for group in glossary.values() if isinstance(group, list) for item in group] if isinstance(glossary, dict) else glossary
    if not isinstance(entries, list):
        return []
    result = []
    for entry in entries[:2000]:
        if not isinstance(entry, dict):
            continue
        original = entry.get('original') or entry.get('source')
        translated = entry.get('translation') or entry.get('translated') or entry.get('target')
        if isinstance(original, str) and isinstance(translated, str):
            result.append({'original': original[:160], 'translation': translated[:240],
                           'aliases': entry.get('aliases', [])})
    return result


def bounded_context(ctx, queries):
    def get(name, default=None):
        return ctx.get(name, default) if isinstance(ctx, dict) else getattr(ctx, name, default)
    context = get('chapter_context', {}) or {}
    if not context:
        path = get('image_name') or get('source_image_path')
        context = load_chapter_context(path) if path else {}
    if not isinstance(context, dict):
        context = {}
    searchable = '\n'.join(queries).casefold()
    terms = glossary_entries(get('custom_prompt_json', {})) + glossary_entries(context)
    relevant = []
    for term in terms:
        names = [term['original']]
        for alias in term.get('aliases', []) if isinstance(term.get('aliases'), list) else []:
            names.append(alias.get('original', '') if isinstance(alias, dict) else str(alias))
        if any(name and str(name).casefold() in searchable for name in names):
            relevant.append({'original': term['original'], 'translation': term['translation']})
    characters = context.get('characters', [])
    characters = characters if isinstance(characters, list) else []
    examples = context.get('approved_examples', [])
    examples = examples if isinstance(examples, list) else []
    regions = get('manga_regions', []) or get('text_regions', []) or []
    voices = []
    for region in regions[:100]:
        if isinstance(region, dict) and region.get('speaker'):
            voices.append({'source': str(region.get('text', ''))[:400],
                          'speaker': str(region['speaker'])[:120],
                          'listener': str(region.get('listener', ''))[:120]})
    result = {'story': str(context.get('story', ''))[:4000], 'glossary': relevant[:80],
              'characters': [{key: str(c.get(key, ''))[:600] for key in ('name', 'voice', 'pronouns')}
                             for c in characters[:30] if isinstance(c, dict)],
              'speakers': voices[:30],
              'approved_examples': [{key: str(e.get(key, ''))[:500] for key in ('source', 'translation')}
                                    for e in examples[-8:] if isinstance(e, dict)]}
    # Bound Unicode bytes, including long glossary strings and character voices.
    while len(json.dumps(result, ensure_ascii=False).encode('utf-8')) > 24_000:
        for key in ('speakers', 'approved_examples', 'characters', 'glossary'):
            if result[key]:
                result[key].pop()
                break
        else:
            result['story'] = result['story'][:1000]
            break
    return result
