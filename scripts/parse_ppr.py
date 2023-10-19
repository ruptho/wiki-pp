import re
from enum import Enum

import mwparserfromhell as mwp
import pandas as pd
import wikichatter.comment as wcu
import wikichatter.indentblock as wib
from mw.lib import title as normalize_title
from wikichatter.signatureutils import NoUsernameError

from scripts.retrieve_ppr import load_pp_req
# "When the wikitext parser keeps misbehaving": https://imgflip.com/i/7g571y
from scripts.util import parse_timestamp, flatten_comment_tree, prune_string, get_day_range_from_string

date_regex = r'([0-3]?[0-9] ([Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|' \
             r'[Oo]ctober|[Nn]ovember|[Dd]ecember) 20[012][0-9])'
EN_DATE_TITLE_REGEX = re.compile(f'\n== ?{date_regex} ?==\n')
EN_ARTICLE_TITLE_REGEX = re.compile(f'\n====?(.+?)?====?\n')

RE_REQ_PROT = re.compile(r"'''([\w -:]+)'''.*")

# For parsing decision
picto_tag = lambda dec_type, ext=True, filetype='svg': \
    f'[[File:Pictogram voting {dec_type}.{filetype}|20px{"|link=|alt=" if ext else ""}]]'
PV_SUPP = picto_tag('support')
PV_DEL = picto_tag('delete')
PV_DECL = picto_tag('oppose')
PV_BLCK = '[[File:Stop x nuvola with clock.svg|20px]]'
PV_KEEP = picto_tag('keep')
PV_COM = picto_tag('comment')
PV_Q = picto_tag('question')
PV_INFO = picto_tag('info')
PV_MVGR = f"{picto_tag('move light green', ext=False, filetype='png')} '''Consider " \
          f"[[WP:AN3|the edit warring noticeboard]]''' " \
          f"– This is a case of possible [[WP:EW|edit-warring]] by one or two users."
RE_PATTERN_DURATION = re.compile(
    r"for a period of '''(.+?)''', after which the page will be automatically unprotected|'''(indefinitely)'''")
RE_PATTERN_PAGELINKS = re.compile(r"\*\W*{{pagelinks\|.+?}}\n")
ALL_PVS = [PV_SUPP, PV_DEL, PV_DECL, PV_BLCK, PV_KEEP, PV_COM, PV_Q, PV_INFO, PV_MVGR]
RE_PATTERN_DECISION = re.compile(f":*({'|'.join([re.escape(pv) for pv in ALL_PVS])})")
PRE_SEMI_PROT = "'''[[Wikipedia:Protection policy#Semi-protection|Semi-protected]]'''"
PRE_PEND_CHANGE = "'''[[Wikipedia:Protection policy#Pending changes protection|" \
                  "Pending-changes protected]]'''"
PRE_FULL_PROT = "'''[[Wikipedia:Protection policy#Full protection|Fully protected]]'''"
PRE_MOVE_PROT = "'''[[Wikipedia:Protection policy#Move protection|Move protected]]'''"
PRE_CREATE_PROT = "'''[[Wikipedia:Protection policy#Creation protection|Creation protected]]'''"
PRE_TEMPLATE_PROT = "'''[[Wikipedia:Protection policy#Template protection|Template protected]]'''"
PRE_EXTENDED = "'''[[Wikipedia:Protection policy#Extended confirmed protection|" \
               "Extended confirmed protected]]'''"
PRE_PROT_LEV = '– No changes to the current protection level are required at this point in time.'
PRE_PAGE_PROT = "– This page isn't currently protected."
PRE_DIS_ACT = "– Not enough recent disruptive activity to [[Wikipedia:Protection policy|justify]]" \
              " protection."
PRE_COLL_DMG = "– Likely collateral damage as one or several users who are making improvements would be" \
               " affected by the requested protection."
PRE_WARN_USR = "– [[Wikipedia:Template messages/User talk namespace|Warn the user appropriately]] then" \
               " report them to [[Wikipedia:Administrator intervention against vandalism|AIV]] or " \
               "[[Wikipedia:Administrators' noticeboard/Incidents|ANI]] if they continue."
PRE_PROT_PREEM = "– [[WP:NO-PREEMPT|Pages are not protected preemptively]]."
PRE_TEMP_WIDE = "– This template is not used widely enough to be considered a " \
                "[[WP:HRT|high-risk template]]."
PRE_DISPUTE = "– Content dispute. Please use the article's talk page or other forms of " \
              "[[Wikipedia:Dispute resolution|dispute resolution]]."
PRE_USR_TLK = "– User talk pages [[Wikipedia:Protection policy#User pages|are not protected]] " \
              "except in response to severe or continued vandalism."
PR_PEND_CHG = "– Pending changes protection [[Wikipedia:Protection policy#When to apply pending " \
              "changes protection|should not be used]] on pages with a high edit rate."
PRE_UNPROT_SOURCED = "– Please create a [[Wikipedia:Reliable sources|sourced]] version of this article " \
                     "in a [[Wikipedia:Subpages|subpage]] or your userspace. When this is done, please " \
                     "make the request again, or ask any [[Wikipedia:Administrators|administrator]] to " \
                     "move the page for you."
PRE_UNPROT_EDITREQ = "– Please use an [[Wikipedia:Edit requests#Making requests|edit request]] to " \
                     "request specific changes to be made to the protected page."
PRE_ALR_UNPROTECTED = "'''Already unprotected'''&nbsp;"
PRE_ALR_PROTECTED = "'''Already protected'''&nbsp;"
PRE_ALR_DONE = " '''Already done'''&nbsp;"
PRE_USER_BLOCK = "'''User(s) [[Wikipedia:Blocking policy|blocked]]'''"
PRE_USER_REBLOCK = "'''User(s) re-[[Wikipedia:Blocking policy|blocked]]'''"

PD_COLUMNS_DATE_SEC = ['date', 'norm_title', 'request_level', 'request_text', 'request_timestamp', 'request_user',
                       'decision_admin', 'decision_status', 'decision_type', 'decision_ts', 'decision_duration',
                       'decision_text']


# === CLASSES
class PPRArchivePage(object):
    def __init__(self, text, title):
        self.title = title
        self.sections_raw = None
        self.wc_page = mwp.parse(text)
        self.raw_text = text
        self.sections = self.parse_ppr_archive_page()

    # https://en.wikipedia.org/w/api.php?action=parse&format=json&page=Wikipedia:Requests_for_page_protection/Archive/2020/03&prop=sections&disabletoc=1
    # https://en.wikipedia.org/w/api.php?action=parse&format=json&page=Wikipedia:Requests_for_page_protection/Archive/2020/03&prop=wikitext&section=1&disabletoc=1
    def parse_ppr_archive_page(self):
        self.split_date_sections()

        sections = []
        print(f'[PARSE] DATE SECTION {self.title} ({len(self.sections_raw)} day sections)')
        for i, section_raw in enumerate(self.sections_raw):
            # section = mwp.parse(section_raw)
            # print(section_raw)
            date_section = PPRArchivePageSection(section_raw)
            if date_section.date_title:
                sections.append(date_section)
        return sections

    def split_date_sections(self):
        # this is unfortunately necessary because mwparserfromhell somehow keeps detecting nested sections
        match_iter = re.finditer(EN_DATE_TITLE_REGEX, self.raw_text)
        indices = [0] + [m.start(0) for m in match_iter] + [len(self.raw_text)]
        self.sections_raw = [self.raw_text[i:j] for i, j in zip(indices, indices[1:])][1:]

    def count_valid_decisions(self):
        n_dates_found, n_dates = len(self.sections), len(get_day_range_from_string(self.title))
        n_articles, n_articles_dec_found = 0, 0
        for date_section in self.sections:
            n_art, n_decs = date_section.count_valid_decisions()
            n_articles, n_articles_dec_found = n_art + n_articles, n_articles_dec_found + n_decs
        return n_dates, n_dates_found, n_articles, n_articles_dec_found

    def find_article_decisions(self):
        print(f'====== [PARSE] Decisions for PAGE {self.title}')
        for date_section in self.sections:
            date_section.find_article_decisions()

    def __str__(self):
        return "<{0}: {1} Days>".format(self.title, len(self.sections))

    def __repr__(self):
        return str(self)

    def simplify(self):
        basic = {"dates": [s.simplify() for s in self.sections]}
        if self.title is not None:
            basic["timeframe"] = self.title
        return basic

    def decisions_to_pandas(self):
        if len(self.sections) > 0:
            df_page = pd.concat([date_section.decisions_to_pandas() for date_section in self.sections])
        else:
            df_page = pd.DataFrame([[pd.NA] * len(PD_COLUMNS_DATE_SEC)], columns=PD_COLUMNS_DATE_SEC)
            print(f'[ERROR] No articles found for {self.title}')
        df_page['archive_title'] = self.title
        return df_page


class PPRArchivePageSection(object):
    def __init__(self, section_wcode: str):
        self.date_title_str = None
        self.date_title = None
        self.level = None
        self._raw_article_sections = []
        self._article_sections = []
        self.raw_text = section_wcode
        self.clean_text = re.sub(RE_PATTERN_PAGELINKS, '', self.raw_text)
        self._wikicode = mwp.parse(section_wcode)
        self.load_section_info()

    def load_section_info(self):
        main_heading = self._wikicode.filter_headings()[0]
        if main_heading.level == 2:
            self.parse_date_heading(main_heading)
            # print(f'{self.date_title}')
            self.split_article_sections()
            print(f'[PARSE] RfPP for {self.date_title} ({len(self._raw_article_sections)} requests)')
            for article_sec_wc in self._raw_article_sections:
                article_sec = PPRArchivePageArticleSection(self.date_title, mwp.parse(article_sec_wc))
                if article_sec.article_title:
                    self._article_sections.append(article_sec)
                else:
                    print(f'No article title {article_sec_wc}')
        else:
            print(f'Invalid Heading for {main_heading.title}')

    def split_article_sections(self):
        # this is unfortunately necessary because mwparserfromhell somehow keeps missing nested sections as separate
        match_iter = re.finditer(EN_ARTICLE_TITLE_REGEX, self.clean_text)
        indices = [0] + [m.start(0) for m in match_iter] + [len(self.clean_text)]
        self._raw_article_sections = [self.clean_text[i:j] for i, j in zip(indices, indices[1:])][1:]

    def parse_date_heading(self, date_heading):
        heading = mwp.parse(date_heading).filter_headings()[0]
        self.level = heading.level
        self.date_title_str = heading.title.strip()
        self.date_title = pd.to_datetime(self.date_title_str).date()

    def find_article_decisions(self):
        # print(f'[PARSE] Decisions for {self.date_title_str} ({len(self.article_sections)} RfPP)')
        i_found = 0
        for article_section in self.article_sections:
            article_section.find_decisions_from_discussion()
            i_found += int(len(article_section.decisions) > 0)
        # print(f'... found decisions for {i_found}/{len(self.article_sections)} RfPP.')

    @property
    def article_sections(self):
        return list(self._article_sections)

    def __str__(self):
        return "<{0}: {1}>".format(self.date_title, len(self._article_sections))

    def __repr__(self):
        return str(self)

    def simplify(self):
        basic = {"articles": [s.simplify() for s in self._article_sections]}
        if self.date_title is not None:
            basic["heading"] = self.date_title_str
            basic["heading_date"] = self.date_title
        return basic

    def decisions_to_pandas(self):
        decisions = []
        if len(self.article_sections):
            for a_s in self.article_sections:
                req = a_s.request
                if req is not None:
                    if not isinstance(req, PPRRequest):
                        print(a_s.norm_title)
                    assert isinstance(req, PPRRequest)
                    req_level, req_text, req_ts, req_user = \
                        req.request_level, req.request_text, req.timestamp, req.user
                else:
                    req_level, req_text, req_ts, req_user = pd.NA, pd.NA, pd.NA, pd.NA
                if len(a_s.decisions) > 0:
                    for dec in a_s.decisions:
                        assert isinstance(dec, PPRDecision)
                        row = [self.date_title, a_s.norm_title, req_level, req_text, req_ts, req_user, dec.admin,
                               dec.status, dec.type, dec.timestamp, dec.duration, dec.decision_text]
                        decisions.append(row)
                else:
                    row = [self.date_title, a_s.norm_title, req_level, req_text, req_ts, req_user] + [pd.NA] * 6
                    decisions.append(row)
        else:
            decisions = [self.date_title] + [pd.NA] * (len(PD_COLUMNS_DATE_SEC) - 1)
            print(f'[ERROR] No articles found for {self.date_title}')
        return pd.DataFrame(decisions, columns=PD_COLUMNS_DATE_SEC)

    def count_valid_decisions(self):
        n_articles, n_dec_found = len(self.article_sections), \
                                  sum(int(len(a_s.decisions) > 0) for a_s in self.article_sections)
        return n_articles, n_dec_found


class PPRArchivePageArticleSection(object):
    def __init__(self, date_title, section_wcode):
        self.date = date_title
        self.title_format = None
        self.level = None
        self._discussion = []
        self._wikicode = section_wcode
        self.article_title = None
        self.title_template = None
        self.norm_title = None
        self.request = None
        self.decisions = None
        self.comments = None
        self.load_article_section_info()

    def load_article_section_info(self):
        wc = self._wikicode
        self.parse_article_title_template()
        indent_blocks = wib.generate_indentblock_list(wc.get_sections(include_headings=False)[-1])
        comments = self.identify_comments_linear_merge_failsafe(indent_blocks)
        self._discussion = comments
        self.comments = list(self._discussion)  # for compatibility
        self.request = PPRRequest(self)

    @staticmethod
    def identify_comments_linear_merge_failsafe(text_blocks):
        working_comment = wcu.Comment()
        comments = [working_comment]
        split_blocks = text_blocks  # []
        # for block in text_blocks:
        #     wc_str = str(block.text).strip()

        #    for t in wc_str.split('\n'):
        #         if len(t.strip()) > 0:
        #            split_blocks.append(wib.IndentBlock(mwp.parse(t.strip()), block.indent + 1))

        for block in split_blocks:
            if working_comment.author is not None:
                working_comment = wcu.Comment()
                comments.append(working_comment)

            try:
                working_comment.add_text_block(block)
            except NoUsernameError as nerr:
                print(f'No valid signature found for {len(block.text)} long comment |{block.text}| ')
        return wcu._sort_into_hierarchy(comments)

    def find_decisions_from_discussion(self):
        # print(f'\t[PARSE] Decision for {self.norm_title}')
        self.decisions = []
        all_comments = flatten_comment_tree(self.discussion)
        for c in all_comments:
            if RE_PATTERN_DECISION.match(prune_string(c.text.replace('\n', ' '))):
                decision = self.parse_decision_status(c)
                # print(decision)
                self.decisions.append(decision)
            else:
                # for debugging
                # print(prune_string(c.text.replace('\n', ' '))[:25])
                pass

    def parse_decision_status(self, comment: wcu.Comment):
        text = prune_string(comment.text.replace('\n', ' '))
        status, dec_type, duration = None, None, None

        def find_accepted_status(txt):
            txt_new, lev = txt, None
            if txt.startswith(PRE_SEMI_PROT):
                txt_new, lev = prune_string(txt, PRE_SEMI_PROT), Decision.SEMI
            elif txt.startswith(PRE_FULL_PROT):
                txt_new, lev = prune_string(txt, PRE_FULL_PROT), Decision.FULL
            elif txt.startswith(PRE_PEND_CHANGE):
                txt_new, lev = prune_string(txt, PRE_PEND_CHANGE), Decision.PEND_CHANGE
            elif txt.startswith(PRE_EXTENDED):
                txt_new, lev = prune_string(txt, PRE_EXTENDED), Decision.EXTENDED
            elif txt.startswith(PRE_MOVE_PROT):
                txt_new, lev = prune_string(txt, PRE_MOVE_PROT), Decision.MOVE
            elif txt.startswith(PRE_CREATE_PROT):
                txt_new, lev = prune_string(txt, PRE_CREATE_PROT), Decision.CREATE
            elif txt.startswith(PRE_TEMPLATE_PROT):
                txt_new, lev = prune_string(txt, PRE_TEMPLATE_PROT), Decision.TEMPLATE
            return txt_new, lev

        def find_declined_status(txt):
            txt_new, lev = txt, None
            if txt.startswith(PRE_PROT_LEV):
                txt_new, lev = prune_string(txt, PRE_PROT_LEV), Decision.DECL_NO_CHANGES
            elif txt.startswith(PRE_PAGE_PROT):
                txt_new, lev = prune_string(txt, PRE_PAGE_PROT), Decision.DECL_UNBLOCK_NOTPROT
            elif txt.startswith(PRE_DIS_ACT):
                txt_new, lev = prune_string(txt, PRE_DIS_ACT), Decision.DECL_ACTIVITY
            elif txt.startswith(PRE_COLL_DMG):
                txt_new, lev = prune_string(txt, PRE_COLL_DMG), Decision.DECL_COLLATERAL
            elif txt.startswith(PRE_WARN_USR):
                txt_new, lev = prune_string(txt, PRE_WARN_USR), Decision.DECL_WARNING
            elif txt.startswith(PRE_PROT_PREEM):
                txt_new, lev = prune_string(txt, PRE_PROT_PREEM), Decision.DECL_PREEMPTIVE
            elif txt.startswith(PRE_TEMP_WIDE):
                txt_new, lev = prune_string(txt, PRE_TEMP_WIDE), Decision.DECL_TEMPLATE
            elif txt.startswith(PRE_DISPUTE):
                txt_new, lev = prune_string(txt, PRE_DISPUTE), Decision.DECL_DISPUTE
            elif txt.startswith(PRE_USR_TLK):
                txt_new, lev = prune_string(txt, PRE_USR_TLK), Decision.DECL_USERTALK
            elif txt.startswith(PR_PEND_CHG):
                txt_new, lev = prune_string(txt, PR_PEND_CHG), Decision.DECL_PENDING_CHANGES
            else:
                lev = Decision.DECL_GENERAL

            return txt_new, lev

        def find_unprotected_status(txt):
            txt_new, lev = txt, None

            if txt.startswith(PRE_UNPROT_SOURCED):
                txt_new, lev = prune_string(txt, PRE_UNPROT_SOURCED), Decision.DECL_NO_CHANGES
            elif txt.startswith(PRE_UNPROT_EDITREQ):
                txt_new, lev = prune_string(txt, PRE_UNPROT_EDITREQ), Decision.DECL_UNBLOCK_NOTPROT
            else:
                lev = Decision.NOT_UNPROT_GENERAL
            return txt_new, lev

        def find_existing_status(txt):
            txt_new, lev = txt, None
            if txt.startswith(PRE_ALR_UNPROTECTED):
                txt_new, lev = prune_string(txt, PRE_ALR_UNPROTECTED), Decision.ALREADY_UNPROTECTED
            elif txt.startswith(PRE_ALR_PROTECTED):
                txt_new, lev = prune_string(txt, PRE_ALR_PROTECTED), Decision.ALREADY_PROTECTED
            elif txt.startswith(PRE_ALR_DONE):
                txt_new, lev = prune_string(txt, PRE_ALR_DONE), Decision.ALREADY_DONE
            return txt_new, lev

        if text.startswith(PV_SUPP):
            status, (text, dec_type) = DecisionClass.PROTECTED, find_accepted_status(prune_string(text, PV_SUPP))
            date_match = RE_PATTERN_DURATION.match(text)
            if date_match is not None:
                groups = date_match.groups()
                duration = (groups[0] if groups[0] is not None else groups[1]).strip()
            text = re.sub(RE_PATTERN_DURATION, '', text)
        elif text.startswith(PV_BLCK):
            text = prune_string(text, PV_BLCK)
            if text.startswith(PRE_USER_BLOCK):
                text, status = prune_string(text, PRE_USER_BLOCK), Decision.USER_BLOCK
            elif text.startswith(PRE_USER_REBLOCK):
                text, status = prune_string(text, PRE_USER_REBLOCK), Decision.USER_REBLOCK
        elif text.startswith(PV_DECL):
            text = prune_string(text, PV_DECL)
            if text.startswith("'''Declined'''"):
                status, (text, dec_type) = DecisionClass.DECLINED, find_declined_status(
                    prune_string(text, "'''Declined'''"))
            elif text.startswith("'''Not unprotected'''"):
                status, (text, dec_type) = DecisionClass.NOT_UNPROTECTED, find_unprotected_status(
                    prune_string(text, "'''Not unprotected'''"))
        elif text.startswith(PV_DEL):
            text = prune_string(text, PV_DEL)
            if text.startswith("'''Not done'''"):
                status, text = DecisionClass.NOT_DONE, prune_string(text, "'''Not done'''")
            elif text.startswith("'''Withdrawn'''"):
                status, text = DecisionClass.WITHDRAWN, prune_string(text, "'''Withdrawn'''")
        elif text.startswith(PV_KEEP):
            status = DecisionClass.HANDLED
            text = prune_string(text, PV_KEEP)
            dec_type, text = Decision.DONE if text.startswith("'''Done'''") else Decision.UNPROTECTED if \
                text.startswith("'''Unprotected'''") else None, prune_string(prune_string(text, "'''Done'''"),
                                                                             "'''Unprotected'''")
        elif text.startswith(PV_INFO):
            status, (text, dec_type) = DecisionClass.EXISTING, (find_existing_status(prune_string(text, PV_INFO)))
        elif text.startswith(PV_COM) or text.startswith(PV_Q):
            # leave this as is for now
            status, text = DecisionClass.COMM_OR_QUESTION, prune_string(prune_string(text, PV_COM), PV_Q)
        elif text.startswith(PV_MVGR):
            status, dec_type, text = DecisionClass.EDIT_WAR, None, prune_string(text, PV_MVGR)
        else:
            print('----- No valid starting string found for comment starting with:')
            print(text[:500])
            print('---------------------')

        return PPRDecision(comment, status, dec_type, duration, text)

    def parse_article_title_template(self):
        # older talk pages
        article_info = self._wikicode.filter_headings()[0]
        self.level = article_info.level

        title_templates = self._wikicode.filter_templates()
        # ==22 May 2015==
        if len(title_templates) > 0 and self.date < pd.to_datetime('22 May 2015').date():
            if len(title_templates[0].params) > 0:
                self.article_title = title_templates[0].params[0]
                self.title_format = f'template-{title_templates[0].name}'
            else:
                self.article_title = title_templates[0].name
                self.title_format = f'template-empty'
        else:
            # newest templates
            title_templates = self._wikicode.filter_wikilinks()

            self.title_format = f'wikilink'
            self.article_title = title_templates[0].title
        self.norm_title = normalize_title.normalize(prune_string(self.article_title, remove_chars=':'))

    @property
    def discussion(self):
        return list(self._discussion)

    def __str__(self):
        return "<{0}: {1}>".format(self.level, self.norm_title)

    def __repr__(self):
        return str(self)

    def simplify(self):
        # basic = {"threads": []}
        # for thread in self._discussion:
        #    basic['threads'].append({'comments': c.simplify() for c in thread})
        basic = {"comments": c.simplify() for c in self._discussion}
        if self.article_title is not None:
            basic["heading"] = self.article_title
        return basic


class PPRRequest(object):
    # Details about the initial comment/request
    def __init__(self, article: PPRArchivePageArticleSection):
        self.top_comment = None
        self.article = article
        self.request_level = None
        self.request_text = None
        self.timestamp = None
        self.user = None
        self.parse_request()

    def parse_request(self):
        self.top_comment = self.article.discussion[0]
        self.timestamp = parse_timestamp(self.top_comment.time_stamp)
        self.user = self.top_comment.author
        self.parse_protection_level_and_test()

    def parse_protection_level_and_test(self):
        comment_text = self.top_comment.text.replace('\n', ' ').strip()
        # comment_text = comment_text.split('\n')[-1] if '\n' in comment_text else comment_text

        if self.user:
            re_match = re.match(build_protection_request_regex(self.user), comment_text)
            if re_match is not None:
                groups_matched = re_match.groups()
                self.request_level, self.request_text = groups_matched[0].strip(), groups_matched[1].strip()
            else:
                re_match = re.match(build_protection_request_regex(self.user, no_prot=True), comment_text)
                if re_match is not None:
                    groups_matched = re_match.groups()
                    self.request_level = None
                    self.request_text = groups_matched[0].strip()
                else:
                    print(f'[ERROR] when parsing with user ({self.user})')
                    print(self.user)
                    self.request_text = comment_text
        else:
            print(f'[PARSE] Without user |{comment_text}|')
            re_match = re.match(build_protection_request_regex(no_user=True), comment_text)
            if re_match is not None:
                groups_matched = re_match.groups()
                self.request_level, self.request_text = groups_matched[0].strip(), groups_matched[1].strip()
            else:
                print('[ERROR] when parsing without user')
                print(comment_text)
                self.request_text = comment_text

        self.request_text = prune_string(self.request_text, remove_chars=' .*:-–', only_left=False)
        if self.request_level is not None and len(self.request_level) > 0:
            self.request_level = self.request_level[:-1] if self.request_level[-1] == ':' else self.request_level
        else:
            self.request_level = 'Unknown'
            # print(f'|{self.request_text}|')

    def __str__(self):
        return f'{self.request_level}|{self.user}|{self.timestamp}|{self.request_text}'

    def __repr__(self):
        return str(self)


def build_protection_request_regex(user='', no_user=False, no_prot=False):
    esc_user = re.escape(user)
    # use only first part, otherwise the second part could lead to problems (user-style formatting, etc.)
    # optional: Check out user regexes in wikichatter.signatureutils
    # As this user has been identified as the comment author using the other regex, so this should suffice!
    if no_user:
        return re.compile(rf".*?'''([\w -:]+)'''(.+)", flags=re.IGNORECASE)
    elif no_prot:
        return re.compile(rf"(.+?)\[\[\W*(Special:Contributions/{esc_user}|User([ _]talk)?:.*?{esc_user}).*?\|",
                          flags=re.IGNORECASE)
    else:
        return re.compile(
            rf".*?'''([\w -:]+)'''(.+?)\[\[\W*(Special:Contributions/{esc_user}|User([ _]talk)?:.*?{esc_user}).*?\|",
            flags=re.IGNORECASE)


def build_decision_user_regex(user):
    esc_user = re.escape(user)
    return re.compile(rf"(.+?)\[\[:?(Special:Contributions/{esc_user}|User([ _]talk)?:{esc_user}) *?\|",
                      flags=re.IGNORECASE)


class PPRDecision(object):
    # information about the final decision
    def __init__(self, comment: wcu.Comment, status, dec_type, duration, text):
        self.status = status
        self.type = dec_type
        self.comment = comment
        self.admin = comment.author
        self.decision_text = text
        self.duration = duration
        self.timestamp_str = comment.time_stamp
        self.timestamp = parse_timestamp(comment.time_stamp)
        if self.admin is not None:
            matched = re.match(build_decision_user_regex(self.admin), self.decision_text)
            if matched is not None:
                self.decision_text = prune_string(matched.groups()[0], remove_chars=".' ", only_left=False)
            else:
                self.decision_text = prune_string(self.decision_text, remove_chars=".' ", only_left=False)
        else:
            self.decision_text = prune_string(self.decision_text, remove_chars=".' ", only_left=False)

    def __str__(self):
        return f'{self.status}|{self.type}|{self.admin}|{self.timestamp}|{self.duration}|{self.decision_text}'

    def __repr__(self):
        return str(self)


class DecisionClass(Enum):
    DECLINED = 'declined'
    PROTECTED = 'protected'
    COMM_OR_QUESTION = 'comment_or_question'
    EXISTING = 'existing'
    EDIT_WAR = 'edit_war'
    HANDLED = 'done_or_unprotected'
    NOT_DONE = 'not_done'
    WITHDRAWN = 'withdrawn'
    NOT_UNPROTECTED = 'not_unprotected'


class Decision(Enum):
    # see: https://en.wikipedia.org/wiki/Template:RFPP
    # Approved protections
    SEMI = 0
    PEND_CHANGE = 1
    FULL = 2
    MOVE = 3
    CREATE = 4
    TEMPLATE = 5
    EXTENDED = 6

    # Declined
    # '''Declined'''
    DECL_GENERAL = 10
    # '''Declined''' – No changes to the current protection level are required at this point in time.
    DECL_NO_CHANGES = 11
    # '''Declined''' – Not enough recent disruptive activity to [[Wikipedia:Protection policy|justify]] protection.
    DECL_ACTIVITY = 12
    # '''Declined''' – Likely collateral damage as one or several users who are making improvements would be affected
    # by the requested protection.
    DECL_COLLATERAL = 13
    # '''Declined''' – [[Wikipedia:Template messages/User talk namespace|Warn the user appropriately]] then report
    # them to [[Wikipedia:Administrator intervention against vandalism|AIV]] or [[Wikipedia:Administrators'
    # noticeboard/Incidents|ANI]] if they continue.
    DECL_WARNING = 14
    # '''Declined''' – [[WP:NO-PREEMPT|Pages are not protected preemptively]].
    DECL_PREEMPTIVE = 15
    # '''Declined''' – This template is not used widely enough to be considered a [[WP:HRT|high-risk template]].
    DECL_TEMPLATE = 16
    # '''Declined''' – Content dispute. Please use the article's talk page or other forms of [[Wikipedia:Dispute
    # resolution|dispute resolution]].
    DECL_DISPUTE = 17
    # '''Declined''' – User talk pages [[Wikipedia:Protection policy#User pages|are not protected]] except in
    # response to severe or continued vandalism.
    DECL_USERTALK = 18
    # '''Declined''' – Pending changes protection [[Wikipedia:Protection policy#When to apply pending changes
    # protection|should not be used]] on pages with a high edit rate.
    DECL_PENDING_CHANGES = 19

    # Unprotection tags
    # '''Declined''' – This page isn't currently protected.
    DECL_UNBLOCK_NOTPROT = 20  # this is also a decline block
    # '''Unprotected'''
    UNPROTECTED = 30
    #  '''Not unprotected'''
    NOT_UNPROT_GENERAL = 31
    # '''Not unprotected''' – Please create a [[Wikipedia:Reliable sources|sourced]] version of this article in a [[
    # Wikipedia:Subpages|subpage]] or your userspace. When this is done, please make the request again, or ask any [[
    # Wikipedia:Administrators|administrator]] to move the page for you.
    NOT_UNPROT_SOURCED = 32
    # '''Not unprotected''' – Please use an [[Wikipedia:Edit requests#Making requests|edit request]] to request
    # specific changes to be made to the protected page.
    NOT_UNPROT_EDITREQ = 33

    # User_blocks
    USER_BLOCK = 40
    USER_REBLOCK = 41

    # Done tags
    DONE = 50
    # '''Already unprotected'''&nbsp;by administrator <span class=""plainlinks"">[
    # //en.wikipedia.org/wiki/User:Example Example]</span>."
    ALREADY_DONE = 51
    # '''Already protected'''&nbsp;by administrator <span class=""plainlinks"">[//en.wikipedia.org/wiki/User:Example
    # Example]</span>."
    ALREADY_UNPROTECTED = 52
    # '''Already done'''&nbsp;by administrator <span class=""plainlinks"">[//en.wikipedia.org/wiki/User:Example
    # Example]</span>."
    ALREADY_PROTECTED = 53

    # Others
    NOTE_OR_QUESTION = 60
    NOTE = 61
    WITHDRAWN = 62
    NOTICEBOARD = 63


def load_and_parse_ppr_archive_pages(save_path):
    pp_req_test = load_pp_req(save_path=save_path)
    return parse_ppr_archive_pages(pp_req_test)


def parse_ppr_archive_pages(pp_req_dict):
    parsed_pages = {}
    for month, page_text in pp_req_dict.items():
        print(f'====== [PARSE] PAGE FOR {month}')
        parsed_pages[month] = PPRArchivePage(page_text, month)
    return parsed_pages
