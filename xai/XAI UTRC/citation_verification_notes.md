# Citation verification notes (web-verified)

[OK] lundberg2017: Authors are exactly two: Scott M. Lundberg and Su-In Lee (confirmed on official NeurIPS proceedings page). The official NeurIPS BibTeX leaves the pages field empty; the page range 4765-4774 comes from DBLP. Official BibTeX booktitle is just "Advances in Neural Information Processing Systems" with volume 30; I rendered booktitle as "Advances in Neural Information Processing Systems 30" per the request (the abstract page itself titles it "Advances in Neural Information Processing Systems 30 (NIPS 2017)"). Editors and publisher taken verbatim from the official NeurIPS BibTeX file.
  evidence: https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html; https://papers.nips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Bibtex.bib

[OK] barkjohn2021: All fields (authors, title, journal, volume 14, issue 6, pages 4617-4637, year 2021, DOI 10.5194/amt-14-4617-2021) confirmed directly on the Copernicus (AMT) publisher page. No uncertainties.
  evidence: https://amt.copernicus.org/articles/14/4617/2021/

[OK] breiman2001: All fields (author, title, journal, volume 45, issue 1, pages 5-32, year 2001, DOI) verified against the Crossref DOI metadata record; Springer Nature Link search result corroborates "Machine Learning 45, 5-32 (2001)". Springer article page itself was behind an auth redirect, so Crossref served as the authoritative source. Crossref lists publisher as "Springer Science and Business Media LLC"; shortened to Springer in the entry.
  evidence: https://api.crossref.org/works/10.1023/A:1010933404324; https://link.springer.com/article/10.1023/A:1010933404324

[OK] ke2017: Authors (8), title, booktitle, editors, publisher, volume, and year confirmed from the official NeurIPS proceedings page and its BibTeX file. The official NeurIPS BibTeX leaves pages empty; pages 3146-3154 are confirmed by DBLP's record for the paper. Address (Long Beach) is standard NIPS 2017 venue info; remove that field if strict page-level evidence is required, as it was not on the fetched pages. No DOI exists for NeurIPS 2017 papers.
  evidence: https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html; https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Bibtex.bib

[OK] prokhorenkova2018: Author list confirmed on the official NeurIPS proceedings page; DBLP lists the first author as "Liudmila Ostroumova Prokhorenkova" but the NeurIPS page uses "Liudmila Prokhorenkova" (used here). Pages 6639-6649 confirmed by DBLP and researchr (some citations erroneously use 6638-6648). Publisher/address/editors omitted as not verified from fetched pages. arXiv version: 1706.09516.
  evidence: https://papers.nips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html; https://dblp.org/rec/conf/nips/ProkhorenkovaGV18.html

[OK] chen2016: Authors, title, proceedings title, pages 785-794, year 2016, publisher ACM, and DOI 10.1145/2939672.2939785 all confirmed via DBLP record and search-result snippets of the ACM DL page. ACM DL blocked direct fetch (HTTP 403), so location/date (San Francisco, Aug 13-17, 2016) were confirmed via the SCIRP-indexed citation and DBLP; the 'address' field reflects the conference location as conventionally cited. The series tag 'KDD '16' is the standard ACM proceedings series designation seen in the DBLP record.
  evidence: https://dblp.org/rec/conf/kdd/ChenG16.html; https://dl.acm.org/doi/10.1145/2939672.2939785

[OK] brey2018: All fields (authors, title, journal, volume 18, issue 3, pages 1745-1761, year 2018, DOI 10.5194/acp-18-1745-2018) verified directly against the Copernicus ACP publisher page. Published 6 February 2018. No discrepancies with the provided reference.
  evidence: https://acp.copernicus.org/articles/18/1745/2018/

[OK] inness2019: All fields for inness2019 confirmed on the Copernicus ACP publisher page (20 authors, vol. 19, pp. 3515-3556, DOI 10.5194/acp-19-3515-2019). Issue number 6 is inferable from the publisher URL path (/articles/19/3515/) but was not explicitly shown as "issue 6" on the fetched page; drop the number field if you want strictly page-evidenced fields only.

Better citation for the CAMS global FORECAST system (what Open-Meteo's air quality API serves): Peuch et al. 2022, verified via Crossref metadata for DOI 10.1175/BAMS-D-21-0314.1:

@article{peuch2022,
  author  = {Peuch, Vincent-Henri and Engelen, Richard and Rixen, Michel and Dee, Dick and Flemming, Johannes and Suttie, Martin and Ades, Melanie and Agust{\'i}-Panareda, Anna and Ananasso, Cristina and Andersson, Erik and Armstrong, David and Barr{\'e}, J{\'e}r{\^o}me and Bousserez, Nicolas and Dominguez, Juan Jose and Garrigues, S{\'e}bastien and Inness, Antje and Jones, Luke and Kipling, Zak and Letertre-Danczak, Julie and Parrington, Mark and Razinger, Miha and Ribas, Roberto and Vermoote, Stijn and Yang, Xiaobo and Simmons, Adrian and Garc{\'e}s de Marcilla, Juan and Th{\'e}paut, Jean-No{\"e}l},
  title   = {The {Copernicus} {Atmosphere} {Monitoring} {Service}: From Research to Operations},
  journal = {Bulletin of the American Meteorological Society},
  volume  = {103},
  number  = {12},
  pages   = {E2650--E2668},
  year    = {2022},
  doi     = {10.1175/BAMS-D-21-0314.1}
}

Recommendation: cite Peuch et al. 2022 for the operational CAMS forecast system served by Open-Meteo; keep Inness et al. 2019 only if the reanalysis dataset itself is used. Flemming et al. 2015 (Geosci. Model Dev., 8, 975-1003, doi:10.5194/gmd-8-975-2015) describes the underlying IFS tropospheric chemistry scheme but its fields were only seen in a search-result snippet, not verified on the publisher page.
  evidence: https://acp.copernicus.org/articles/19/3515/2019/; https://api.crossref.org/works/10.1175/BAMS-D-21-0314.1

[OK] zippenfenig: Citation matches exactly what open-meteo.com/en/licence requests: Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. DOI 10.5281/zenodo.7970649 is the concept DOI covering all versions; it resolves on Zenodo to the latest release (v1.4.0, published 2024-12-31, CC BY 4.0). The site's requested year is 2023, so 2023 is used even though newer versioned records (e.g. 10.5281/zenodo.14582479, 2024) exist; cite a version-specific DOI only if a particular release must be pinned. In IEEE style this renders as: P. Zippenfenig, "Open-Meteo.com Weather API," Zenodo, 2023, doi: 10.5281/zenodo.7970649.
  evidence: https://open-meteo.com/en/licence; https://zenodo.org/doi/10.5281/zenodo.7970649

[OK] ejscreen: Verified against the actual PDF title page (downloaded and text-extracted): title "EJScreen Technical Documentation for Version 2.3", dated July 31, 2024, U.S. EPA, Office of Environmental Justice and External Civil Rights, Washington, D.C. 20460. PDF metadata Author field confirms "United States Environmental Protection Agency". The document's own suggested citation is "U.S. Environmental Protection Agency (EPA), 2024. EJScreen Technical Documentation." Version 2.3 (July 2024) is the latest and final version, superseding the v2.2 (July 2023) mentioned in the task hint. EJScreen was removed from the live EPA site on Feb. 5, 2025 (per EDGI); the standard citable source now is the EPA "19january2025snapshot" archive, whose EJScreen index page confirms v2.3 as current. Note the tool's subtitle on the tech-doc title page is "Environmental Justice Mapping and Screening Tool" (the archived web page uses "Screening and Mapping"); I omitted the subtitle from the BibTeX title since the document title itself does not include it. No DOI exists for this document. Alternative mirror if the snapshot URL breaks: Public Environmental Data Partners reconstruction at https://screening-tools.com/epa-ejscreen.
  evidence: https://19january2025snapshot.epa.gov/system/files/documents/2024-07/ejscreen-tech-doc-version-2-3.pdf; https://19january2025snapshot.epa.gov/ejscreen/index.html

[OK] choubin2025: Exact paper confirmed. Crossref record for DOI 10.1016/j.rineng.2025.105976 gives: authors Bahram Choubin, Abolfazl Jaafari, Jalal Henareh, Omid Karimi, Farzaneh Sajedi Hosseini; journal Results in Engineering (Elsevier), volume 27, article number 105976, print date September 2025, online June 25, 2025. ScienceDirect page (S2590123025020481) returned 403 to direct fetch, but the article appeared in web search results with matching title/journal; all fields taken from Crossref, the publisher-deposited registry. "105976" is an article number (used in the pages field per common Elsevier practice); no traditional page range exists.
  evidence: https://api.crossref.org/works/10.1016/j.rineng.2025.105976; https://www.sciencedirect.com/science/article/pii/S2590123025020481

[OK] meyer2022: BOTH papers exist and were confirmed; details below.

(A) Nature Communications 2022 — CONFIRMED via Crossref API (10.1038/s41467-022-29838-9) and the PMC full text (PMC9033849).
  Meyer, Hanna; Pebesma, Edzer. "Machine learning-based global maps of ecological variables and the challenge of assessing them." Nature Communications 13(1), article number 2208, 2022. ISSN 2041-1723. Publisher: Springer Nature.
  Note on "pages": this is an article-number journal; 2208 is the article number, not a page range. I put it in `pages` because that is the conventional BibTeX rendering for Nature Communications; if your IEEE style prefers, use `number = {2208}` as article no. and drop the issue. No true page range exists — do not invent one.

(B) Methods in Ecology and Evolution 2021 — CONFIRMED via Crossref API (10.1111/2041-210X.13650). The Wiley landing page itself returned HTTP 403 to my fetcher, so Crossref is the evidence of record; the search result listing for the Wiley page corroborated title/year.
  @article{meyer2021aoa,
    author  = {Meyer, Hanna and Pebesma, Edzer},
    title   = {Predicting into unknown space? Estimating the area of applicability of spatial prediction models},
    journal = {Methods in Ecology and Evolution},
    year    = {2021},
    volume  = {12},
    number  = {9},
    pages   = {1620--1633},
    issn    = {2041-210X},
    doi     = {10.1111/2041-210X.13650}
  }

WHICH ONE SUPPORTS THE CLAIM "random cross-validation overstates spatial prediction skill; leave-location-out validation is required":
  Use (A), the 2022 Nature Communications paper. It is the one that makes this argument directly. Verified quotes from the PMC full text: naive random n-fold or leave-one-out CV "makes sense when the data are independent and identically distributed. When this is not the case, dependencies between nearby samples ... are ignored and result in biased, overly optimistic model assessment," and "spatial cross-validation ... that control for such dependencies are the only way to overcome this bias."
  The 2021 MEE paper (B) is about the area-of-applicability / dissimilarity-index concept — where a model may validly be applied — not primarily about random-CV optimism. Cite it only if your sentence is about the area of applicability.

CAVEAT on the exact phrase "leave-location-out": the 2022 paper argues for spatial cross-validation generally. The specific LLO/LTO/LLTO terminology and the empirical demonstration that random k-fold vastly overstates skill (R^2 0.90 random vs 0.24 LLO for Antarctic air temperature) come from a different paper, confirmed via Crossref (10.1016/j.envsoft.2017.12.001):
  Meyer, H.; Reudenbach, C.; Hengl, T.; Katurji, M.; Nauss, T. "Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation." Environmental Modelling & Software, vol. 101, pp. 1-9, 2018. ISSN 1364-8152.
  If your IEEE paper's sentence literally says "leave-location-out validation is required," the most defensible citation is a pair: [meyer2022] for the overstatement claim + Meyer et al. 2018 for LLO specifically. Citing meyer2022 alone for the LLO term is a slight stretch.

Fields I could NOT confirm and therefore omitted: no page range for (A) (article-number journal); no month fields for any entry; I did not verify author middle names or affiliations beyond what Crossref/PMC list.
  evidence: https://api.crossref.org/works/10.1038/s41467-022-29838-9; https://pmc.ncbi.nlm.nih.gov/articles/PMC9033849/

[OK] desouza2021: All fields verified against Crossref metadata for DOI 10.1038/s41370-021-00328-2; title and journal also corroborated by the Nature.com article listing in search results. Published online May 6, 2021; print issue May 2021 (vol. 31, issue 3, pp. 514-524). The Nature page itself redirects to an auth gateway so the Crossref registry record was used as the authoritative source.
  evidence: https://api.crossref.org/works/10.1038/s41370-021-00328-2; https://www.nature.com/articles/s41370-021-00328-2

[OK] ruminski2006: All fields verified: title and full author list (Ruminski, Kondragunta, Draxler, Zeng) extracted directly from the paper PDF on EPA's server; conference name, location (New Orleans, LA, Wyndham Canal Place), dates (May 15-18, 2006), and session (Session 10, Managed Burning and Wildfires, presented May 18) verified from the official final program PDF. The program cover says '15th Annual Emission Inventory Conference' while EPA elsewhere calls it the '15th International Emission Inventory Conference' - both names are used interchangeably. No page numbers, volume, or DOI exist for this venue (papers are posted individually without pagination), so those fields are intentionally omitted. The alternate title mentioned ('Use of multiple satellite sensors in NOAA's operational near real-time fire and smoke detection and characterization program') is a DIFFERENT paper: Ruminski et al., Proc. SPIE 7089, 70890A (2008), DOI 10.1117/12.807507 - it is 2008, not 2006, so it should not carry the ruminski2006 key.
  evidence: https://gaftp.epa.gov/Air/nei/ei_conference/EI15/session10/ruminiski.pdf; https://gaftp.epa.gov/Air/nei/ei_conference/EI15/finalprogram.pdf
