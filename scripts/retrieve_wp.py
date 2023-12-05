import pandas as pd
import requests as rq

from scripts.util import extract_rfpp_from_wikitable

headers = {"User-Agent": "custom_address@mail.com"} # change this accordingly

MW_API = lambda wiki: f"https://{wiki}.wikipedia.org/w/api.php"


# action=query&prop=revisions&titles=Wikipedia:Requests_for_page_protection/Archive/2012/10&rvslots=*&rvprop=content&formatversion=2
# action=query&formatversion=2&titles=Wikipedia%3ARequests+for+page+protection%2FArchive%2F2012%2F10&slots=%2A&prop=revision&rvprop=content&rvslots=%2A
def get_page_content(pagetitle, language='en'):
    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "titles": pagetitle,
        "rvprop": 'content',
        "rvslots": '*',
        "formatversion": "2",
        "format": "json"
    }

    data = rq.get(url=MW_API(language), params=PARAMS, headers=headers).json()
    page = data["query"]["pages"][0]

    rev_content = None
    try:
        if len(page['revisions']) > 1:
            print('More than 1 revision?? ERROR!')
        for revision in page["revisions"]:
            rev_data = revision['slots']['main']['content']
            rev_content = rev_data
    except:
        print(f'Error for {pagetitle}')
        rev_content = f'Error when retrieving {pagetitle}'
    return rev_content


def get_expanded_template(template, language='en'):
    PARAMS = {
        "action": "expandtemplates",
        "prop": "wikitext",
        "text": template,
        "format": "json"
    }

    data = rq.get(url=MW_API(language), params=PARAMS, headers=headers).json()
    return data["expandtemplates"]["wikitext"]


def get_page_contents(pagetitles, language):
    return [get_page_content(pagetitle, language) for pagetitle in pagetitles]

def retrieve_expanded_templates_from_file(path='doc/rfpp_templates.txt',
                                          out_path='doc/rfpp_templates_expanded.txt'):
    retrieved = retrieve_expanded_templates(extract_rfpp_from_wikitable(path))
    if out_path:
        retrieved.to_csv(out_path, index=False, sep='\t')
    return retrieved


def retrieve_expanded_templates(templates):
    results = []
    for tmp_text in templates:
        results.append([tmp_text, get_expanded_template(tmp_text)])
    return pd.DataFrame(results, columns=['template', 'expanded'])
