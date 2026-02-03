# Seminar Guidelines

## 1\) Purpose

A seminar should demonstrate your ability to work independently using scientific methods on a problem from your study subject within a specified period. It should be of the quality of an (applied) scientific paper that can be submitted to an ML conference or workshop.

## 2\) Prerequisites

* **ML Basics:** content equivalent to Chapters 1–10 of the [Introduction to ML](https://slds-lmu.github.io/i2ml/) lecture.  
* **Software/Programming:** proficiency in R or Python; routine use of Git.  
* **When interpretable ML is relevant:** Self-study the interpretable ML topics relevant to your topic using, e.g., the [online videos](https://slds-lmu.github.io/iml/) taught in the “Advanced ML” lecture, the corresponding chapters from the “[Interpretable ML](https://christophm.github.io/interpretable-ml-book/)” book, and the papers linked in both resources.

## 3\) Recommended Guides

* Story and narrative: [Writing a good scientific paper](https://medium.com/@black_51980/writing-a-good-scientific-paper-c0f8af480c91)  
* Structure: [How to write a research paper](http://lucalongo.eu/howtowritearesearchpaper.html)  
* Abstracts: [Nature Journal’s abstract writing tips](https://www.nature.com/documents/nature-summary-paragraph.pdf)  
* Best practices: [ICML’s paper writing best practices](https://icml.cc/Conferences/2022/BestPractices)   
* How to   
  * [do literature research](https://docs.google.com/presentation/d/1O1b5aqCNhJfuy0V0VTc8FuS153Cg16LkRpWChH38rBc/edit#slide=id.g29163c13401_0_42),   
  * [develop a research hypothesis](http://researchdesign.lucalongo.eu/material/RESEARCH_DESIGN___developing_research_hypothesis.pdf),   
  * use [simulation studies to evaluate statistical methods](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8086?s=09),   
  * [make nice tables](https://people.inf.ethz.ch/markusp/teaching/guides/guide-tables.pdf),   
  * and give [good presentations](https://docs.google.com/presentation/d/1-YSTCv2m5V3NTB2q4mKif6qCaMzqCujYi3cKPIQL_ZY/edit?usp=drive_link)

## 4\) Tools

* **Overleaf:** use the provided template ([alternative template](https://www.overleaf.com/latex/templates/unofficial-lmu-slds-thesis-template/phzzzfvsyphv) if none was provided).  
* **GitHub:** create a repository and invite your supervisor; you may connect Overleaf to GitHub.  
* **Reference management:** Zotero recommended.  
* See [here](https://docs.google.com/presentation/d/1ofz8sx56KzL1a8bNRjYs5-W-4aPaoFynHyv5dkv-gEU/edit?slide=id.g29257f41d17_1_26#slide=id.g29257f41d17_1_26) for a list of some more useful tools for project management, etc.

## 5\) Deliverables and Scope

* Focus on novel methods, findings, and key results. Do not re-explain well-known basics (e.g., random forest internals) unless essential for your contribution.  
* Code for experiments should be stored in a GitHub repository shared with your supervisor.   
* Ensure the code is fully reproducible, well-documented, and follows a style guide (links below):  
  * Checklist [for writing reproducible code](https://onlinelibrary.wiley.com/pb-assets/assets/15214036/RR_Guideline.pdf)  
  * Make sure the code you provide is clean and follows a unified style guide, e.g., for R code:  
    * [https://github.com/mlr-org/mlr3/wiki/Style-Guide](https://github.com/mlr-org/mlr3/wiki/Style-Guide)  
    * [http://adv-r.had.co.nz/Style.html](http://adv-r.had.co.nz/Style.html)  
  * Include the GitHub link in your report, as this is done in scientific papers  
* Use of AI tools: follow the policy [Use of AI-Tools in Academic Works](https://docs.google.com/document/u/0/export?format=pdf&id=1mQ5YAWKDt9Ax4LvK-wuUzO5XfCI0e-ZGdk49ghb08Ds&token=AC4w5VieJuEx0tuFggMm1u6gzVv4Itn2nA%3A1759775842016&ouid=101380224888923735088&includes_info_params=true&cros_files=false&tab=t.0).  
* Protocols: Use the **[Progress Meeting \[YOUR NAME\]](https://docs.google.com/document/d/1Dyqr-kR_dDW99E_IMDbEDeqEa3irgFKMLdkyfqHeU2M/edit?usp=sharing)** template to structure and prepare meetings with your supervisor (or team meetings if you want), document our discussions, and continuously track your progress.  
* Presentation duration:  
  * Single speaker: 30 min talk \+ 10 min discussion led by the discussant (total 40 min).  
  * Team: 35–40 min talk \+ 5 \- 10 min discussion.  
  * Respect time limits with 10 percent tolerance.  
* Report page limits (excluding references and appendix):  
  * BA: 15–30 pages. MA: 20–40 pages.  
  * It is acceptable to aim for the lower bound (if you don’t exaggerate with dozens of figures).  
  * Your study regulations are binding regarding the minimum and maximum length and presentation duration (please double-check this).

## 6\) Roles: Discussant and Reviewer

* **Provide feedback (slides \+ report) on another student’s topic (before deadlines)**  
  * Requires understanding the main paper topic well enough  
  * Write a review of your colleague’s presentation and report (max. 3 pages, see also “[How to review a paper](https://natolambert.com/guides/how-to-review-a-paper)”, “[Step by step guide to reviewing a manuscript](https://authorservices.wiley.com/Reviewers/journal-reviewers/how-to-perform-a-peer-review/step-by-step-guide-to-reviewing-a-manuscript.html)”, and [this](https://taliaringer.wordpress.com/2023/08/14/reviewing-papers-with-adhd/)) and deliver substantial, actionable feedback well before the talk and report deadline so your colleague has time to revise  
    \-\> Include these max. 3 pages as an appendix to your own report.  
  * Coordinate timelines proactively (e.g., make sure you receive a current draft of slides/report latest 1 week before deadline and deliver your feedback 3 days before deadline); report any delays and issues to your supervisor  
* **Lead the post-talk discussion**  
  * Prepare at least two substantive prompts (questions or discussion points)  
  * Open the Q\&A and moderate/facilitate the discussion (invite others, connect related points, keep the discussion moving)

## 7\) Teamwork (if applicable)

* Collaborative working on GitHub mainly for R/Python code:  
  * Versioning: Version all code from day one; each member commits under their own account.  
  * Use GitHub as intended (for collaborative work): manage tasks/todos via issues, develop on feature branches with pull requests that clearly summarize their content, and ensure code quality through peer review and documented decisions  
  * Modular and reproducible code: structure code into meaningful subfolders and ensure full reproducibility from scripts or configuration files.  
  * Contribution clarity: by project end, it must be obvious who did what (commits, PRs, issues, or code comments).  
* Report Contributions:  
  * In each section, highlight who the main author (listed first) was and who only provided reviews and minor edits. Multiple main authors per section are allowed, but ensure a balanced mix of individually authored and jointly authored sections.   
  * Include a dedicated section for each team member’s contributions. Each author should clearly describe their role in the project and report, specifying the tasks they performed. For reference, follow the style of "Author Contributions" sections commonly found in [PhD dissertations](https://edoc.ub.uni-muenchen.de/view/subjects/fak16.referee_one_name.html#group_Bischl,_Bernd), typically placed before the appended research papers inside the dissertation.  
  * Overall, contributions must be fairly distributed among all team members.

## 8\) Assessment Overview

Use the following as guidance rather than a strict checklist. Approximate weights reflect typical emphasis.

### 8.1 Scientific Presentation (about 30–40 percent)

* Balanced speaking time across group members.  
* See “Evaluation Sheet Presentations” on Moodle.

### 8.2 Final Report (about 60–70 percent)

* Research quality: clear questions, motivation, originality, relevance.  
* Literature review: focused and critical; identifies gaps that motivate your work.  
* Methodology: appropriate, rigorous, and justified; correct mathematics; sound implementation.  
* Technical competence: efficient experiments, clean code, good engineering practice.  
* Analysis and evaluation: thorough experiments, valid statistics, insightful interpretation, explicit limitations.  
* Reproducibility: documented code and data, repeatable pipelines, verification or validation steps.  
* Writing and presentation: clear structure, high-quality figures, proper citations; effective oral communication.  
* Project management and impact: timely, independent work; meaningful contribution with publishable potential.  
* Collaboration quality and discussant performance also matter.