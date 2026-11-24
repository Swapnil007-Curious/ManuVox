# Writing refinement and IEEE version: list of edits (2026-11-24)

This file lists every sentence changed in the paper's prose, in two passes made in the same
commit. No number was changed: every number is still a `\V{...}` macro generated from the result
files, and `scripts/paper_claims.py` re-checks all of them (0 failures). No citation key was changed
or removed.

1. The writing refinement changed wording only. A script compared the committed text with the
   edited text and found the same macros, citations, references, labels and significance words
   in every section, so no claim was strengthened or weakened.
2. The fixes for the third, independent review (`REVIEW.md`, items R3 to R21) go further, and
   these sentences are the ones to read with care. They soften claims the reviewer found stronger
   than the evidence: "in part because the serving pipeline does not reproduce the training
   features" (abstract), "largely appearance-free", "(weak evidence about new signers)", "one had a
   large effect in both pipelines" (was "a large and consistent effect"), the parity test "checks"
   (was "makes") that Python and the browser compute the same inputs, "no evidence that the rest of
   the gap depends on which of the two pipelines runs" (was "on the pipeline"), "No threshold or
   constant was fitted on these clips" (was "Nothing was tuned"), and the v3 sentence "is
   consistent with ... but an in-domain score cannot settle it" (was "points to"). They add
   statements the reviewer asked for, each supported by the code or the result files: which
   discrepancies are parity questions, that clip scores do not use the abstention threshold, that
   browser clip scores move in thirds, the Holm family sizes (two macros that `paper_tables.py`
   already generated, `mde.py.family` and `mde.web.family`), the definition of the ASL Citizen
   signer interval, the runtime of the quantization latencies, and a clarified power statement.
   To keep the full version's main text at eight pages, three passages moved unchanged to the
   appendix (the "other changes" of the live-gap ablation, the details of the compromised v3 Teach
   Dex run, the service worker sentence), each with a pointer from the main text; Part 1 shows
   them as removed in one place and added in the other. British spellings became American
   (license, favor, analyzed), as IEEE asks.

What the writing refinement does: split long sentences at semicolons where the two halves are separate claims;
replace a double negative ("was not untouched"), "serve only to" and other indirect verbs with
plain ones; put the actor first where it matters ("We chose the variants and synonyms with AI
assistance ..."); remove the one bold phrase in the text (System Demonstration); replace a
sentence about the paper ("This paper is a measurement study of ...") with one about its subject;
rename a slogan-like paragraph heading ("One specification, one test." became "Shared
specification and parity test."); rephrase an aphorism in Related Work and the awkward "the GRU is
-1.5 points lower". Contrasts that carry information were kept, for example "not two training
procedures", "inconclusive rather than absent" and "in-domain and not a fair comparison with v2".

What was checked and needed no change: the paper has no em dashes; its en dashes are number
ranges (`\V{a}--\V{b}`) or compounds of two equal terms ("train--serve",
"convolution--transformer"), which are not connectors and were kept. None of the words to avoid
(delve, crucial, pivotal, robust, showcase, leverage, utilize, landscape, tapestry, testament,
underscore, highlight, intricate, meticulous, seamless, furthermore, additionally, notably,
moreover) occurs; `scripts/paper_claims.py` now fails if one appears. The sources contain no
zero-width characters, BOM, soft hyphens or non-breaking spaces outside `~`, and every quotation
uses LaTeX quotes. Section headings keep title case, as both the ACL and the IEEE style require.

Generated text: one table caption changed (`scripts/paper_tables.py`, the live-gap ablation): "a
dash marks a configuration that exists only in the browser. Clip scores do not use the abstention
threshold." was added after "Browser rows are means over three real-time runs". The other table
captions were read and left as they are. The IEEE tables carry a short title above each table
(new text, listed in `IEEE_TITLES` in `scripts/paper_tables.py`) and the full caption as a note.

Non-prose changes in the same commit: PDF metadata (`\hypersetup`: title and author only;
xdvipdfmx itself still writes Producer and CreationDate), `\hyphenation{web-cam}`, a thicker
border on the audited boxes of the pipeline figure, and in `references.bib` two titles braced so
that BibTeX keeps their capitals ("{E}valuating" in ovadia2019shift, the Kaggle competition's
name) plus the new entry for the extended version (`shukla2026manuvox`).

## Part 1: full version (`paper/main.tex`, `paper/sections/`), 62 changes

Each item gives the sentence(s) as committed before (git HEAD) and after the edit.
### `paper/main.tex`

1. Before: (none: added)

   After: \hyphenation{web-cam web-cams} \hypersetup{pdftitle={ManuVox: Auditing the Train\textendash Serve Gap of a Landmark-Based Sign Recognizer on a Laptop}, pdfauthor={Swapnil Shukla}, pdfcreator={}, pdfproducer={}}

2. Before: (none: added)

   After: \input{sections/acknowledgment}

### `paper/sections/abstract.tex`

3. Before: Landmark-based sign recognizers can score well on a benchmark split and still fail on a live webcam, because the serving pipeline does not reproduce the training features.

   After: Landmark-based sign recognizers can score well on a benchmark split and still fail on a live webcam, in part because the serving pipeline does not reproduce the training features.

4. Before: On our signer-independent split of the Google ISLR data, the first deployment's CNN reaches \V{m.cnn.top1}\% top-1, yet it recognized \V{lg.web.c0headbuild.top1}\% of \V{lg.clips} new webcam clips from one signer; a convolution--transformer hybrid reaches \V{m.d.top1}\% and classifies a sign in \V{b.wasm1.p50} ms (median) in Chrome.

   After: On our signer-independent split of the Google Isolated Sign Language Recognition (ISLR) data, the first deployment's CNN reaches \V{m.cnn.top1}\% top-1, yet it recognized \V{lg.web.c0headbuild.top1}\% of \V{lg.clips} new webcam clips from one signer. A convolution--transformer hybrid, v2, reaches \V{m.d.top1}\% and classifies a sign in \V{b.wasm1.p50} ms (median) in Chrome.

5. Before: With all pre-specified fixes, shared through one feature specification and a golden parity test, the hybrid's clip top-1 reaches \V{lg.web.c6final.top1}\% in the browser and \V{lg.py.c6final.top1}\% with the training tracker, closing \V{lg.web.recover}--\V{lg.py.recover}\% of the gap between its unfixed live score and its test accuracy; one fix, a motion segmenter, if anything hurt.

   After: With all pre-specified fixes, shared through one feature specification and a golden-fixture test, the hybrid's clip top-1 reaches \V{lg.web.c6final.top1}\% in the browser and \V{lg.py.c6final.top1}\% with the training tracker, closing \V{lg.web.recover}--\V{lg.py.recover}\% of the gap between its unfixed live score and its test accuracy. One fix, a motion segmenter, if anything hurt.

6. Before: A second model, v3, trained on ASL Citizen as well and with webcam augmentation, reaches \V{cit.v3.top1}\% there, an in-domain score, and gains \V{v3c.ks.diff} points on Google ISLR, but live it gains \V{v3c.web.diff} and \V{v3c.py.diff} points, not significant with fifty clips.

   After: A second model, v3, also trained on ASL Citizen, with webcam augmentation, reaches \V{cit.v3.top1}\% there (in-domain) and gains \V{v3c.ks.diff} points on Google ISLR, but live it gains \V{v3c.web.diff} and \V{v3c.py.diff} points, not significant with fifty clips.

### `paper/sections/acknowledgment.tex`

7. Before: (none: added)

   After: the Acknowledgment section, with the exact text the author supplied (`paper/sections/acknowledgment.tex`, also input by the IEEE version).

### `paper/sections/appendix.tex`

8. Before: Mirroring flips \V{data.mirroredpct}\% of the Google ISLR sequences, but it is a property of signers rather than sequences: \V{mir.high} of the \V{data.signers} signers have more than 80\% of their sequences mirrored and \V{mir.low} have under 5\%, consistent with left-handed signers or mirrored (selfie) recordings, which the data cannot tell apart; every test signer is at most \V{mir.testmax}\%.

   After: Mirroring flips \V{data.mirroredpct}\% of the Google ISLR sequences, but it is a property of signers rather than sequences: \V{mir.high} of the \V{data.signers} signers have more than 80\% of their sequences mirrored and \V{mir.low} have under 5\%, consistent with left-handed signers or mirrored (selfie) recordings, which the data cannot tell apart. Every test signer is at most \V{mir.testmax}\%.

9. Before: (none: added)

   After: \paragraph{The compromised v3 run of Teach Dex.} Its model-alone score on take 2, \V{td.v3.model}\%, is far below v3's \V{lg3.web.v3.take2}\% on the same clips in the live runs of Section~\ref{sec:liveresults} (for v2 the two agree, \V{td.v2.model}\% and \V{lg3.web.v2.take2}\%), and its runs found ever fewer segments (\V{td.v3.segmax} down to \V{td.v3.segmin}, against \V{td.v2.segmin}--\V{td.v2.segmax} with v2), for a reason the harness, which logs no frame rate, cannot show. Clips without a segment cannot change, so v3's gain (\V{td.v3.diff} points, $p$ = \V{td.v3.p}) is biased toward zero and says nothing about teaching v3.

10. Before: (none: added)

   After: \paragraph{Other changes.} Swapping in the hybrid before any fix gained \V{lg.web.c1model.dtop1} points in the browser (\V{lg.web.c1model.dtop1better} clips better, none worse; \V{lg.web.c1model.dtop1sig} at clip level, not at sign level, Holm $p$ = \V{lg.web.c1model.dtop1psign}) and \V{lg.py.c1model.dtop1} in Python, from a floor near zero. Resampling by time (d), which cannot matter in Python, changed browser clip top-1 by \V{lg.web.c4time.dtop1} points, and calibration (f), which cannot change a segment's top-1, by \V{lg.web.c6final.dtop1}. Replacing the hybrid by the v1 CNN in the final pipeline cost \V{lg.py.xv1model.dtop1} and \V{lg.web.xv1model.dtop1} points (\V{lg.web.xv1model.dtop1sig}), so the hybrid's test-set advantage shows only weakly here.

11. Before: The live clips and both datasets are not redistributed, and neither is v3, which ASL Citizen's licence keeps private.

   After: The live clips and both datasets are not redistributed, and neither is v3, which ASL Citizen's license keeps private. The web app is a static site: a service worker caches the model for offline use and adds the cross-origin isolation headers that WebAssembly threads need on static hosting.

### `paper/sections/conclusion.tex`

12. Before: Of six candidate train--serve discrepancies, one had a large and consistent effect, putting each hand in the wrong slot; a shared feature specification and a golden parity test now make Python and the browser compute the same model inputs from the same landmarks. With the fixes, the hybrid reaches \V{lg.web.c6final.top1}\% clip top-1 in the browser and \V{lg.py.c6final.top1}\% with the training tracker, with no evidence that the rest of the gap depends on the pipeline.

   After: Of six candidate train--serve discrepancies, one had a large effect in both pipelines, putting each hand in the wrong slot. A shared feature specification and a golden-fixture test now check that Python and the browser compute the same model inputs from the same landmarks. With the fixes, the hybrid reaches \V{lg.web.c6final.top1}\% clip top-1 in the browser and \V{lg.py.c6final.top1}\% with the training tracker, with no evidence that the rest of the gap depends on which of the two pipelines runs.

13. Before: Thresholds chosen on dataset sequences need live validation data; next come an oracle segmentation of these clips and live recordings from many Deaf signers, made with their involvement.

   After: Thresholds chosen on dataset sequences need live validation data. The next steps are an oracle segmentation of these clips and live recordings from many Deaf signers, made with their involvement.

### `paper/sections/data.tex`

14. Before: We use the Google ISLR data \citep{kaggle2023islr} (licence CC BY 4.0): \V{data.total} sequences of MediaPipe Holistic landmarks for \V{data.signs} isolated ASL signs, recorded by \V{data.signers} signers.

   After: We use the Google ISLR data \citep{kaggle2023islr} (license CC BY 4.0): \V{data.total} sequences of MediaPipe Holistic landmarks for \V{data.signs} isolated ASL signs, recorded by \V{data.signers} signers.

15. Before: Our split assigns signers whole, by a seeded random permutation, to training (\V{data.trainsigners} signers, \V{data.trainseq} sequences), validation (\V{data.valsigners}, \V{data.valseq}) and test (\V{data.testsigners}, \V{data.testseq}); it is not an official split.

   After: Our split assigns signers whole, by a seeded random permutation, to training (\V{data.trainsigners} signers, \V{data.trainseq} sequences), validation (\V{data.valsigners}, \V{data.valseq}) and test (\V{data.testsigners}, \V{data.testseq}). It is not an official split.

16. Before: The test split was not untouched during development: the v2 features, recipe and architecture were designed after the v1 models had been evaluated on it, and the model to ship was chosen by test accuracy (Section~\ref{sec:models}).

   After: The test split was used during development: the v2 features, recipe and architecture were designed after the v1 models had been evaluated on it, and the model to ship was chosen by test accuracy (Section~\ref{sec:models}).

17. Before: ASL Citizen \citep{desai2023aslcitizen} holds \V{cit.all.videos} webcam videos of \V{cit.all.glosses} glosses by \V{cit.all.signers} signers with official signer-disjoint splits (\V{cit.all.trainsigners}, \V{cit.all.valsigners} and \V{cit.all.testsigners} signers); its licence allows non-commercial research and forbids redistribution.

   After: ASL Citizen \citep{desai2023aslcitizen} holds \V{cit.all.videos} webcam videos of \V{cit.all.glosses} glosses by \V{cit.all.signers} signers with official signer-disjoint splits (\V{cit.all.trainsigners}, \V{cit.all.valsigners} and \V{cit.all.testsigners} signers). Its license allows non-commercial research and forbids redistribution.

18. Before: The variant and synonym choices were made with AI assistance from ASL-LEX's English translations and are not confirmed by a signer; they are part of the method and a possible source of error.

   After: We chose the variants and synonyms with AI assistance from ASL-LEX's English translations, and no signer has confirmed them; they are part of the method and a possible source of error.

19. Before: The training and validation videos (\V{v3.cit.train} and \V{v3.cit.val}) serve only to train v3.

   After: The training and validation videos (\V{v3.cit.train} and \V{v3.cit.val}) are used only to train v3.

### `paper/sections/efficiency.tex`

20. Before: Post-training quantization of the v1 models (Appendix~\ref{app:quant}) \citep{jacob2018quantization,nagel2021whitepaper} halves every file in fp16 at no measurable cost in accuracy. int8 always costs accuracy (\V{q.int8.least} to \V{q.int8.most} points across variants), and dynamic int8 speeds up the transformer but slows the depthwise-convolution models (hybrid p50 \V{q.hybrid.int8dynamic.p50} vs.\ \V{q.hybrid.onnxfp32.p50} ms).

   After: Post-training quantization of the v1 models (Appendix~\ref{app:quant}) \citep{jacob2018quantization,nagel2021whitepaper} halves every file in fp16 at no measurable cost in accuracy. int8 always costs accuracy (\V{q.int8.least} to \V{q.int8.most} points across variants). In Python onnxruntime, dynamic int8 speeds up the transformer but slows the depthwise-convolution models, the hybrid included (p50 \V{q.hybrid.int8dynamic.p50} ms against \V{q.hybrid.onnxfp32.p50} ms in fp32).

21. Before: v3 has the same architecture and file size (\V{lat.v3.mb}\,MB); in a later benchmark of three interleaved rounds per model its median p50 and p95 were \V{lat.v3.p50} and \V{lat.v3.p95} ms, against \V{lat.v2.p50} and \V{lat.v2.p95} for v2. Both had slow rounds (p95 from \V{lat.p95lo} to \V{lat.p95hi} ms over all rounds), so we see no difference beyond round-to-round variation; all are inside the \V{budget.ms} ms budget.

   After: v3 has the same architecture and file size (\V{lat.v3.mb}\,MB). In a later benchmark of three interleaved rounds per model, its median p50 and p95 were \V{lat.v3.p50} and \V{lat.v3.p95} ms, against \V{lat.v2.p50} and \V{lat.v2.p95} for v2. Both had slow rounds (p95 from \V{lat.p95lo} to \V{lat.p95hi} ms over all rounds), so we see no difference beyond round-to-round variation, and all are inside the \V{budget.ms} ms budget.

22. Before: The classifier took \V{t.hudon.cls1} and \V{t.hudon.cls2} ms per segment (p50, two runs), and a word appeared \V{t.hudon.e2e1} ms (p50) after the segmenter ended the sign; that decision comes \V{seg.lostms}--\V{seg.stillms} ms after the hands leave or stop, so a user waits about \V{t.userlo}--\V{t.userhi} ms.

   After: The classifier took \V{t.hudon.cls1} and \V{t.hudon.cls2} ms per segment (p50, two runs), and a word appeared \V{t.hudon.e2e1} ms (p50) after the segmenter ended the sign. That decision comes \V{seg.lostms}--\V{seg.stillms} ms after the hands leave or stop, so a user waits about \V{t.userlo}--\V{t.userhi} ms.

### `paper/sections/ethics.tex`

23. Before: This paper is an engineering study of one failure mode, the train--serve gap, and not a claim of usefulness to the people the technology is meant for.

   After: This paper studies one engineering failure mode, the train--serve gap; it makes no claim that the system is useful to the people the technology is meant for.

24. Before: The public site's Practice mode does neither yet; it carries only the general note that ManuVox is not a full translator.

   After: The public site's Practice mode does none of this yet; it carries only the general note that ManuVox is not a full translator.

25. Before: ASL Citizen's videos show the people who recorded them for that dataset; we use them only as its licence allows, for non-commercial research, keep the landmarks we extracted and every per-video prediction private, report aggregate results only, keep the model trained on them off the public site, and will delete the derived data when the research ends.

   After: ASL Citizen's videos show the people who recorded them for that dataset; we use them only as its license allows, for non-commercial research, keep the landmarks we extracted and every per-video prediction private, report aggregate results only, keep the model trained on them off the public site, and will delete the derived data when the research ends.

26. Before: The app processes video on the device and keeps taught examples there; when the user asks for a sentence, the recognized glosses, and nothing else, are sent to a third-party language model provider, whose retention policy we have not audited; a gloss sequence can reveal what a user said.

   After: The app processes video on the device and keeps taught examples there. When the user asks for a sentence, the recognized glosses, and nothing else, are sent to a third-party language model provider, whose retention policy we have not audited; a gloss sequence can reveal what a user said.

### `paper/sections/introduction.tex`

27. Before: Landmark-based recognizers fit: MediaPipe keypoints \citep{lugaresi2019mediapipe} are small, appearance-free and fast to classify, and the Google ISLR data \citep{kaggle2023islr} provides them for \V{data.signs} isolated signs, collected with the partners behind PopSign, a game that helps hearing parents of deaf children learn to sign \citep{starner2023popsign}; the vocabulary is everyday and child-oriented.

   After: Landmark-based recognizers suit this setting: MediaPipe keypoints \citep{lugaresi2019mediapipe} are small, largely appearance-free and fast to classify. The Google ISLR data \citep{kaggle2023islr} provides them for \V{data.signs} isolated signs, collected with the partners behind PopSign, a game that helps hearing parents of deaf children learn to sign \citep{starner2023popsign}; the vocabulary is everyday and child-oriented.

28. Before: Any mismatch is train--serve skew \citep{sculley2015debt,breck2017mltestscore}, and it is invisible to offline evaluation.

   After: Any mismatch is train--serve skew \citep{sculley2015debt,breck2017mltestscore}, and offline evaluation cannot see it.

29. Before: This paper is a measurement study of that gap for ManuVox, our in-browser recognizer, on a low-power laptop.

   After: We measure that gap for ManuVox, our in-browser recognizer, on a low-power laptop.

30. Before: \item A signer-independent comparison of landmark models with paired, Holm-corrected tests (Section~\ref{sec:models}): a convolution--transformer hybrid and a longer recipe improve test top-1 by \V{cmp.ship.diff} points over a CNN, for each of the \V{data.testsigners} test signers, while richer input features alone did not help measurably; on an Intel i3-1115G4 laptop the hybrid classifies a sign in \V{b.wasm1.p50} ms in Chrome (Section~\ref{sec:efficiency}).

   After: \item A signer-independent comparison of landmark models with paired, Holm-corrected tests (Section~\ref{sec:models}): a convolution--transformer hybrid and a longer recipe improve test top-1 by \V{cmp.ship.diff} points over a CNN, for each of the \V{data.testsigners} test signers (weak evidence about new signers), while richer input features alone did not help measurably. On an Intel i3-1115G4 laptop the hybrid classifies a sign in \V{b.wasm1.p50} ms in Chrome (Section~\ref{sec:efficiency}).

31. Before: v3, trained also on ASL Citizen and with webcam augmentation, reaches \V{cit.v3.top1}\% there, an in-domain score, with \V{v3c.ks.diff} points on the Google ISLR test split; live, it gains \V{v3c.web.diff} points of clip top-1 in the browser and \V{v3c.py.diff} in Python, neither significant with fifty clips (Section~\ref{sec:liveresults}). \item A personalization feature, Teach Dex, evaluated separately (Section~\ref{sec:teach}): examples replayed from one recording per sign raise clip top-1 on a second take from \V{td.v2.model}\% to \V{td.v2.teach}\% with v2, in the best case of one signer and one sitting; a v3 run of the same test was compromised by its harness.

   After: v3, trained also on ASL Citizen and with webcam augmentation, reaches \V{cit.v3.top1}\% there, an in-domain score, and gains \V{v3c.ks.diff} points on the Google ISLR test split. Live, it gains \V{v3c.web.diff} points of clip top-1 in the browser and \V{v3c.py.diff} in Python, neither significant with fifty clips (Section~\ref{sec:liveresults}). \item A personalization feature, Teach Dex, evaluated separately (Section~\ref{sec:teach}): examples replayed from one recording per sign raise clip top-1 on a second take from \V{td.v2.model}\% to \V{td.v2.teach}\% with v2, in the best case of one signer and one sitting. A v3 run of the same test was compromised by its harness.

32. Before: One feature specification shared by Python and the browser, and a golden-fixture test between them, now keep the two pipelines from drifting apart (Section~\ref{sec:faults}).

   After: One feature specification shared by Python and the browser, and a golden-fixture test between them, now guard against the two pipelines drifting apart (Section~\ref{sec:faults}).

### `paper/sections/limitations.tex`

33. Before: If the signer's productions differ from the dataset's signers, some live errors are production differences rather than pipeline or camera effects, and nothing in this design separates them; a rating of each clip by a fluent, ideally Deaf, signer would.

   After: If the signer's productions differ from the dataset's signers, some live errors are production differences rather than pipeline or camera effects, and nothing in this design separates them. A rating of each clip by a fluent, ideally Deaf, signer would.

34. Before: How the \V{lg.signs} signs were chosen was not recorded; if they favour signs the author found easy to learn, the live estimate is biased.

   After: How the \V{lg.signs} signs were chosen was not recorded; if they favor signs the author found easy to learn, the live estimate is biased.

35. Before: Teach Dex was taught and tested on two takes recorded by one signer in one sitting, with examples replayed from one production per sign, which is its best case; its v3 run is compromised (Section~\ref{sec:teach}) and needs repeating with a harness that logs its frame rate. The live clips are not an independent test of v3 either: they were analysed before v3's webcam augmentation was designed, that augmentation targets the discrepancies they revealed, and Teach Dex reuses them.

   After: Teach Dex was taught and tested on two takes recorded by one signer in one sitting, with examples replayed from one production per sign, which is its best case. Its v3 run is compromised (Section~\ref{sec:teach}) and needs repeating with a harness that logs its frame rate. The live clips are not an independent test of v3 either: they were analyzed before v3's webcam augmentation was designed, that augmentation targets the discrepancies they revealed, and Teach Dex reuses them.

### `paper/sections/livegap.tex`

36. Before: We audited six candidate train--serve discrepancies, fixed each so that the browser reproduces Python (Figure~\ref{fig:pipeline}), and measured each fix on new recordings; not every candidate turned out to be a fault.

   After: We audited six candidate train--serve discrepancies, fixed each so that the browser reproduces Python (Figure~\ref{fig:pipeline}), and measured each fix on new recordings. Not every candidate turned out to be a fault. Three are parity questions, where serving must reproduce what training did: (a), (c) and (d). Mirroring (b) is a counterfactual, segmentation (e) has no counterpart in training, whose sequences each hold one sign, and calibration (f) cannot change a segment's top-1.

37. Before: \caption{The live pipeline in the browser. The orange boxes, marked (a)--(f), are the six audited steps; each reads its constants from \cfg{features\_spec.json} or the model metadata, which the Python training code also reads.}

   After: \caption{The live pipeline in the browser. The orange boxes with thick borders, marked (a)--(f), are the six audited steps; \emph{Holistic layout} is the landmark layout of the training tracker. Each step reads its constants from \cfg{features\_spec.json} or the model metadata, which the Python training code also reads.}

38. Before: The shipped build already detected on the unmirrored frame; detection on the mirrored frame, as a selfie pipeline would do it, is a counterfactual.

   After: The shipped build already detected on the unmirrored frame. Detection on the mirrored frame, as a selfie pipeline would run it, is a counterfactual.

39. Before: The shipped build corrected this with a body-ratio estimate over the first two seconds; the fix estimates the training data's width-to-height ratio from face geometry (\V{asp.train}) and rescales live y about the shoulder midpoint by $k = r_{\mathrm{train}} / (w/h)$, with the frame's real width and height.

   After: The shipped build corrected this with a body-ratio estimate over the first two seconds; the fix estimates the training data's width-to-height ratio $r_{\mathrm{train}}$ from face geometry (\V{asp.train}) and rescales live y about the shoulder midpoint by $k = r_{\mathrm{train}} / (w/h)$, with the frame's real width and height.

40. Before: \paragraph{One specification, one test.}

   After: \paragraph{Shared specification and parity test.}

41. Before: A golden-fixture test in continuous integration runs the browser code on the raw landmarks of \V{par.cases} validation sequences exported from Python and requires every model input to match within \V{par.tol} and the top-1 class to agree; it covers hand assignment and replays a stream through the segmenter.

   After: A golden-fixture test in continuous integration runs the browser code on the raw landmarks of \V{par.cases} validation sequences exported from Python and requires every model input to match within \V{par.tol} and the top-1 class to agree. It also covers hand assignment and replays a stream through the segmenter.

42. Before: The author recorded \V{lg.clips} clips with a laptop webcam: \V{lg.signs} everyday signs from the vocabulary (Table~\ref{tab:persign}), learned from online ASL video dictionaries, two takes each (median \V{lg.duration}\,s), unmirrored at \V{lg.width}$\times$\V{lg.height} and about \V{lg.fps} frames per second (Figure~\ref{fig:skeleton}), scaled to \V{lg.ywidth}$\times$\V{lg.yheight} for both pipelines. The author is both the developer and the only signer, and a pilot of the same person motivated the order of the fixes (Appendix~\ref{app:pilot}). Nothing was tuned on these clips: every threshold and constant comes from the training and validation splits, and the analysis was fixed in the configuration before the first run (the repository cannot show this, as configuration and results were committed together).

   After: The author recorded \V{lg.clips} clips with a laptop webcam: \V{lg.signs} everyday signs from the vocabulary (Table~\ref{tab:persign}), learned from online ASL video dictionaries, two takes each (median \V{lg.duration}\,s). The clips are unmirrored, \V{lg.width}$\times$\V{lg.height} at about \V{lg.fps} frames per second (Figure~\ref{fig:skeleton}), and scaled to \V{lg.ywidth}$\times$\V{lg.yheight} for both pipelines. The author is both the developer and the only signer, and a pilot with the same person motivated the order of the fixes (Appendix~\ref{app:pilot}). No threshold or constant was fitted on these clips: every threshold and constant comes from the training and validation splits, and the analysis was fixed in the configuration before the first run (the repository cannot show this, as configuration and results were committed together).

43. Before: One browser per clip runs every configuration in a seeded random order, three times each (Table~\ref{tab:livegapci}), and a clip's score is its mean over runs; in \V{lg.stalled} of \V{lg.browserrecords} browser records the tracker processed fewer than half of the frames, kept as measured.

   After: One browser per clip runs every configuration in a seeded random order, three times each (Table~\ref{tab:livegapci}), and a clip's score is its mean over runs, so a browser clip can change by a third. In \V{lg.stalled} of \V{lg.browserrecords} browser records the tracker processed fewer than half of the frames; these records are kept as measured.

44. Before: A clip's prediction is the mean probability vector of its classified segments; a clip without one counts as wrong. We report clip top-1 and top-3 with percentile bootstrap intervals \citep{efron1994bootstrap} and compare each configuration with the one it modifies by a paired sign-flip test on per-clip scores \citep{dror2018hitchhiker}, Holm-corrected within each pipeline and metric \citep{dror2017replicability}; intervals are not adjusted. This clip-level analysis was specified before the runs; as a sign's two takes are not independent, a sensitivity analysis added afterwards treats each sign as one unit. Power is low: significance at the first Holm step needs at least \V{mde.web.k} clips (\V{mde.web.pts} points) moving in one direction in the browser and \V{mde.py.k} in Python, so smaller effects are inconclusive rather than absent \citep{card2020power}.

   After: A clip's prediction is the mean probability vector of its classified segments; a clip without one counts as wrong, and the abstention threshold does not enter clip scores. We report clip top-1 and top-3 with 95\% percentile bootstrap intervals over clips \citep{efron1994bootstrap} and compare each configuration with the one it modifies by a paired sign-flip test on per-clip scores \citep{dror2018hitchhiker}, Holm-corrected within each pipeline and metric \citep{dror2017replicability} (\V{mde.py.family} top-1 comparisons in Python, \V{mde.web.family} in the browser). Intervals are not adjusted. This clip-level analysis was specified before the runs. Because a sign's two takes are not independent, a sensitivity analysis added afterwards treats each sign as one unit. Power is low: significance at the first Holm step needs at least \V{mde.web.k} clips moving in one direction in the browser (\V{mde.web.pts} points if each moves by a whole clip) and \V{mde.py.k} in Python, so smaller effects are inconclusive rather than absent \citep{card2020power}.

45. Before: Removing all aspect correction from the final pipeline cost \V{lg.py.xaspectoff.dtop1} points in Python and \V{lg.web.xaspectoff.dtop1} in the browser, and replacing presence segmentation by the motion segmenter cost \V{lg.py.c5segmenter.dtop1} and \V{lg.web.c5segmenter.dtop1}; none of these top-1 changes is significant after correction at either unit, so we treat them as inconclusive.

   After: Removing all aspect correction from the final pipeline cost \V{lg.py.xaspectoff.dtop1} points in Python and \V{lg.web.xaspectoff.dtop1} in the browser, and in the cumulative ablation replacing presence segmentation by the motion segmenter cost \V{lg.py.c5segmenter.dtop1} and \V{lg.web.c5segmenter.dtop1}. None of these top-1 changes is significant after correction at either unit, so we treat them as inconclusive.

46. Before: Swapping in the hybrid before any fix gained \V{lg.web.c1model.dtop1} points in the browser (\V{lg.web.c1model.dtop1better} clips better, none worse; \V{lg.web.c1model.dtop1sig} at clip level, not at sign level, Holm $p$ = \V{lg.web.c1model.dtop1psign}) and \V{lg.py.c1model.dtop1} in Python, from a floor near zero. Resampling by time (d), which cannot matter in Python, changed browser clip top-1 by \V{lg.web.c4time.dtop1} points, and calibration (f), which cannot change a segment's top-1, by \V{lg.web.c6final.dtop1}. Replacing the hybrid by the v1 CNN in the final pipeline cost \V{lg.py.xv1model.dtop1} and \V{lg.web.xv1model.dtop1} points (\V{lg.web.xv1model.dtop1sig}), so the hybrid's test-set advantage shows only weakly here.

   After: Appendix~\ref{app:livegap} describes the remaining rows of Table~\ref{tab:livegap}: the hybrid before any fix, time resampling, calibration, and the v1 CNN in the final pipeline, which shows the hybrid's test-set advantage only weakly.

47. Before: We see no evidence that the training tracker separates these clips from the test split, but the intervals allow a sizeable advantage for Holistic, and the pipelines also differ in real-time frame dropping.

   After: We see no evidence that the training tracker does better on these clips, but the intervals allow a sizeable advantage for Holistic, and the pipelines also differ in real-time frame dropping.

48. Before: \V{lg.failboth} of the \V{lg.signs} signs fail in every take in both pipelines, and for \V{lg.sharedconf} the most frequent wrong guess is the same in both (\V{lg.sharedconflist}; Table~\ref{tab:persign}); a guess such as sleepy for sleep may be the same sign under another label.

   After: In both pipelines, \V{lg.failboth} of the \V{lg.signs} signs fail in every take, and for \V{lg.sharedconf} the most frequent wrong guess is the same in both (\V{lg.sharedconflist}; Table~\ref{tab:persign}). A guess such as sleepy for sleep may be the same sign under another label.

49. Before: Both differences favour v3, but neither is significant with fifty clips: v3's added data and augmentation narrowed this signer's gap far less than they raised its in-domain ASL Citizen score, if at all.

   After: Both differences favor v3, but neither is significant with fifty clips: v3's added data and augmentation narrowed this signer's gap far less than they raised its in-domain ASL Citizen score, if at all.

50. Before: That v3 recognizes \V{cit.v3.top1}\% of the videos of ASL Citizen's unseen signers, trimmed to the sign, but under half of these clips points to segmentation or this signer's productions rather than to webcam conditions as such.

   After: v3 recognizes \V{cit.v3.top1}\% of the videos of ASL Citizen's unseen signers, trimmed to the sign, but under half of these clips. That is consistent with segmentation or this signer's productions, more than webcam conditions as such, limiting the live score, but an in-domain score cannot settle it.

51. Before: With v2, clip top-1 on take 2 rose from \V{td.v2.model}\% \V{td.v2.modelci} to \V{td.v2.teach}\% \V{td.v2.teachci}, \V{td.v2.diff} points \V{td.v2.ci} (\V{td.v2.better} clips better, \V{td.v2.worse} worse; Holm $p$ = \V{td.v2.p}, \V{td.v2.sig}); in a control that taught only \V{td.v2.half} of the signs, the untaught signs changed by \V{td.v2.harm} points.

   After: With v2, clip top-1 on take 2 rose from \V{td.v2.model}\% \V{td.v2.modelci} to \V{td.v2.teach}\% \V{td.v2.teachci}, \V{td.v2.diff} points \V{td.v2.ci} (\V{td.v2.better} clips better, \V{td.v2.worse} worse; Holm $p$ = \V{td.v2.p}, \V{td.v2.sig}). In a control that taught only \V{td.v2.half} of the signs, the untaught signs changed by \V{td.v2.harm} points.

52. Before: The v3 run of this test is compromised: its model-alone score on take 2, \V{td.v3.model}\%, is far below v3's \V{lg3.web.v3.take2}\% on the same clips in the live runs above (for v2 the two agree, \V{td.v2.model}\% and \V{lg3.web.v2.take2}\%), and its runs found ever fewer segments (\V{td.v3.segmax} down to \V{td.v3.segmin}, against \V{td.v2.segmin}--\V{td.v2.segmax} with v2), for a reason the harness, which logs no frame rate, cannot show. Clips without a segment cannot change, so v3's gain (\V{td.v3.diff} points, $p$ = \V{td.v3.p}) is biased toward zero and says nothing about teaching v3.

   After: The v3 run of this test is compromised by its harness and says nothing about teaching v3 (Appendix~\ref{app:v3}).

### `paper/sections/models.tex`

53. Before: Among the v1 models, the transformer's difference from the CNN is \V{cmp.tf.sig} and the GRU is \V{cmp.gru.diff} points lower (Table~\ref{tab:comparisons}).

   After: Among the v1 models, the transformer's difference from the CNN is \V{cmp.tf.sig} and the GRU differs from it by \V{cmp.gru.diff} points (Table~\ref{tab:comparisons}).

54. Before: The hybrid step also multiplies the parameter count by \V{m.paramratio} and the training time by \V{m.timeratio}, so its gain is not the architecture's alone.

   After: The hybrid step also multiplies the parameter count by \V{m.paramratio} and the training time by \V{m.timeratio}, so its gain cannot be credited to the architecture alone.

55. Before: Per signer, the hybrid's top-1 ranges from \V{sig.d.min}\% to \V{sig.d.max}\% (Table~\ref{tab:persigner}); it beats the CNN for all four test signers, weak evidence about new ones, and the recipe's small gain is positive for all four, which four signers cannot make significant.

   After: Per signer, the hybrid's top-1 ranges from \V{sig.d.min}\% to \V{sig.d.max}\% (Table~\ref{tab:persigner}). It beats the CNN for all four test signers, which is weak evidence about new ones; the recipe's small gain is also positive for all four, but four signers cannot make it significant.

56. Before: The shipping rule took the highest \emph{test} top-1 with a p95 browser latency under \V{budget.ms} ms and a file under ten megabytes (Figure~\ref{fig:acclat} in the appendix); the size cap, a repository convenience, alone excludes the ensemble. Because the rule used the test split, we note that validation top-1 selects the same model (\V{m.d.val}\% against at most \V{m.c.val}\% for the others).

   After: The shipping rule took the highest \emph{test} top-1 with a p95 browser latency under \V{budget.ms} ms and a file under ten megabytes (Figure~\ref{fig:acclat} in the appendix). The size cap, a repository convenience, alone excludes the ensemble. Because the rule used the test split, we also checked validation top-1, which selects the same model (\V{m.d.val}\% against at most \V{m.c.val}\% for the others).

57. Before: Across datasets the calibration degrades: on ASL Citizen the error after scaling is \V{cit.v2.ece}; the threshold keeps its precision (\V{cit.v2.sel}\% selective accuracy) but, as accuracy falls, accepts only \V{cit.v2.cov}\% of videos. \paragraph{A second training set (v3).}

   After: Across datasets the calibration degrades: on ASL Citizen the error after scaling is \V{cit.v2.ece}. The threshold keeps its precision (\V{cit.v2.sel}\% selective accuracy) but, as accuracy falls, accepts only \V{cit.v2.cov}\% of videos. \paragraph{Training on a second dataset (v3).}

58. Before: On ASL Citizen, v2 reaches \V{cit.v2.top1}\% (signer interval \V{cit.v2.top1cis}), \V{cit.v2.drop} points below its own test split, and less on signs mapped through a variant (\V{cit.v2.variant}\%) than on exact matches (\V{cit.v2.exact}\%), which may reflect the mapping or different sign forms. v3 reaches \V{cit.v3.top1}\%, but it has seen ASL Citizen's training signers, so that score is in-domain, not a fair contest.

   After: On ASL Citizen, v2 reaches \V{cit.v2.top1}\% (95\% bootstrap interval over signers \V{cit.v2.top1cis}), \V{cit.v2.drop} points below its own test split, and less on signs mapped through a variant (\V{cit.v2.variant}\%) than on exact matches (\V{cit.v2.exact}\%), which may reflect the mapping or different sign forms. v3 reaches \V{cit.v3.top1}\%, but it has seen ASL Citizen's training signers, so that score is in-domain and not a fair comparison with v2.

### `paper/sections/related.tex`

59. Before: ASL Citizen evaluates only on signers unseen in training and targets webcam-based dictionary lookup, the setting closest to ours; we follow its signer-disjoint protocol, test on its official test split, and train a second model on part of its training split.

   After: ASL Citizen evaluates only on signers unseen in training and targets webcam-based dictionary lookup, the setting closest to ours. We follow its signer-disjoint protocol, test on its official test split and train a second model on part of its training split.

60. Before: \citet{moryossef2021pose} show what the skeletal representation loses with two pose estimators: a model is only as good as its serving-time landmarks.

   After: \citet{moryossef2021pose} show with two pose estimators what the skeletal representation loses, so the landmarks available at serving time limit what a model can recognize.

61. Before: Faults accumulate unnoticed in the code around a deployed model \citep{sculley2015debt}, and production checklists test that training and serving compute the same features \citep{breck2017mltestscore}; live input also differs in distribution from the dataset, and our live gap mixes both. Temperature scaling corrects miscalibration, which the expected calibration error summarizes \citep{guo2017calibration}; selective prediction abstains below a threshold \citep{geifman2017selective}, as our ``not sure'' does.

   After: Faults accumulate unnoticed in the code around a deployed model \citep{sculley2015debt}, and production checklists test that training and serving compute the same features \citep{breck2017mltestscore}. Live input also differs in distribution from the dataset, and our live gap mixes skew and shift. Temperature scaling corrects miscalibration, which the expected calibration error summarizes \citep{guo2017calibration}. Selective prediction abstains below a threshold \citep{geifman2017selective}, as our ``not sure'' does.

### `paper/sections/system.tex`

62. Before: On request, a serverless proxy sends only the recognized glosses to a hosted language model (\cfg{llama-3.1-8b-instant} on Groq), instructed to write one short sentence without adding facts, and the browser speaks it; offline, a rule-based joiner produces a sentence marked as basic. \textbf{Sentence generation is not evaluated}; a fluent sentence can lend false authority to wrong glosses and omits the grammar ASL carries in space, movement and the face. Video and taught examples never leave the device, and a service worker caches the model for offline use and adds the cross-origin isolation headers that WebAssembly threads need on static hosting. The public site serves v2; a build option selects v3 for local research only, as ASL Citizen's licence requires.

   After: On request, a serverless proxy sends only the recognized glosses to a hosted language model (\cfg{llama-3.1-8b-instant} on Groq), instructed to write one short sentence without adding facts, and the browser speaks it. Offline, a rule-based joiner produces a sentence marked as basic. Sentence generation is not evaluated. A fluent sentence can lend false authority to wrong glosses and omits the grammar ASL carries in space, movement and the face. Video and taught examples never leave the device (Appendix~\ref{app:repro} describes the hosting). The public site serves v2; a build option selects v3 for local research only, as ASL Citizen's license requires.

## Part 2: IEEE version (`paper/ieee/sections/`), 77 sentences

The IEEE version is new. Its abstract, introduction, conclusion and Acknowledgment are the files
of the full version (Part 1 covers them). Its other sections are condensed from the edited full
version: most sentences are copied unchanged, some are left out (they remain in the full version,
which the IEEE version cites as its extended version), and the sentences below were shortened,
merged or written for this version. Each item gives the closest sentence of the full version and
the IEEE sentence. Differences of citation command only (`\citep` against `\cite`) are not
listed.


### `paper/ieee/sections/data.tex`

1. Full version: We drop sequences with fewer than \V{data.minhand} frames in which a hand is detected (\V{data.excluded} sequences), leaving \V{data.kept}; each sign keeps between \V{data.signmin} and \V{data.signmax} sequences.

   IEEE version: We drop sequences with fewer than \V{data.minhand} frames in which a hand is detected (\V{data.excluded} sequences), leaving \V{data.kept}.

2. Full version: Checkpoint selection, early stopping, temperature fitting and the abstention threshold use validation only, and a single script reads the test split and logs every run (Table~\ref{tab:testaccess}).

   IEEE version: Checkpoint selection, early stopping, temperature fitting and the abstention threshold use validation only, and a single script reads the test split and logs every run.

3. Full version: ASL Citizen \citep{desai2023aslcitizen} holds \V{cit.all.videos} webcam videos of \V{cit.all.glosses} glosses by \V{cit.all.signers} signers with official signer-disjoint splits (\V{cit.all.trainsigners}, \V{cit.all.valsigners} and \V{cit.all.testsigners} signers).

   IEEE version: ASL Citizen \cite{desai2023aslcitizen} holds \V{cit.all.videos} webcam videos of \V{cit.all.glosses} glosses by \V{cit.all.signers} signers with official signer-disjoint splits.

4. Full version: We chose the variants and synonyms with AI assistance from ASL-LEX's English translations, and no signer has confirmed them; they are part of the method and a possible source of error.

   IEEE version: We chose the variants and synonyms with AI assistance from ASL-LEX's English translations, and no signer has confirmed them.

5. Full version: The official test split, \V{cit.videos} videos of \V{cit.signs} signs by \V{cit.signers} signers (\V{cit.excluded} more had too few hand frames), is read by a second logged script once per model: for v2 it is a cross-dataset test.

   IEEE version: The official test split, \V{cit.videos} videos of \V{cit.signs} signs by \V{cit.signers} signers, is read by a second logged script once per model: for v2 it is a cross-dataset test.

6. Full version: Face and lip points and z are not used, which keeps the trackers' different depth scales out of the model.

   IEEE version: Face and lip points and z are not used.

7. Full version: Frames without a hand are removed; coordinates are centered on the median shoulder midpoint and scaled by the median shoulder distance; a sequence whose left hand is present in more frames than its right is mirrored (x\,$\to$\,1$-$x with the hand blocks and left/right pose points swapped; Appendix~\ref{app:mirror}); and each sequence is linearly resampled to \V{data.frames} frames.

   IEEE version: Frames without a hand are removed; coordinates are centered on the median shoulder midpoint and scaled by the median shoulder distance; a sequence whose left hand is present in more frames than its right is mirrored; and each sequence is linearly resampled to \V{data.frames} frames.

8. Full version: The \emph{v1} input keeps both hands and \V{data.v1pose} pose points (nose, shoulders, elbows, wrists), \V{data.v1points} landmarks in all.

   IEEE version: The \emph{v1} input keeps both hands and \V{data.v1pose} pose points, \V{data.v1points} landmarks in all.

### `paper/ieee/sections/efficiency.tex`

9. Full version: Post-training quantization of the v1 models (Appendix~\ref{app:quant}) \citep{jacob2018quantization,nagel2021whitepaper} halves every file in fp16 at no measurable cost in accuracy. int8 always costs accuracy (\V{q.int8.least} to \V{q.int8.most} points across variants).

   IEEE version: Post-training quantization of the v1 models \cite{jacob2018quantization,nagel2021whitepaper} halves every file in fp16 at no measurable cost in accuracy. int8 always costs accuracy (\V{q.int8.least} to \V{q.int8.most} points across variants).

10. Full version: In Python onnxruntime, dynamic int8 speeds up the transformer but slows the depthwise-convolution models, the hybrid included (p50 \V{q.hybrid.int8dynamic.p50} ms against \V{q.hybrid.onnxfp32.p50} ms in fp32).

   IEEE version: In Python onnxruntime, dynamic int8 speeds up the transformer but slows the depthwise-convolution models, the hybrid included (p50 \V{q.hybrid.int8dynamic.p50} ms against \V{q.hybrid.onnxfp32.p50} ms in fp32), so we ship the hybrid in fp16 (\V{m.ship.mb}\,MB).

11. Full version: One hybrid call takes \V{q.hybrid.fp16.p50} ms (p50) in Python onnxruntime with one thread and \V{b.wasm1.p50} ms (p95 \V{b.wasm1.p95}) in onnxruntime-web with WebAssembly SIMD and one thread; two and four threads (\V{b.wasm2.p50} and \V{b.wasm4.p50} ms) and WebGPU on the integrated GPU (\V{b.webgpu1.p50} ms) are slower.

   IEEE version: One hybrid call takes \V{b.wasm1.p50} ms (p50; p95 \V{b.wasm1.p95}) in onnxruntime-web with WebAssembly SIMD and one thread; more threads and WebGPU on the integrated GPU are slower.

12. Full version: The selection benchmark behind Table~\ref{tab:models} (fp32, p50 \V{m.d.p50} ms for the hybrid) differs from these runs by a few milliseconds.

   IEEE version: The selection benchmark behind Table~\ref{tab:models} (fp32) differs from this run by a few milliseconds, and v3, with the same architecture, showed no difference from v2 beyond round-to-round variation.

13. Full version: With a pilot clip (Appendix~\ref{app:pilot}) as camera, tracking ran at \V{t.hudon.fps} frames per second with the hand skeleton drawn and \V{t.hudoff.fps} without it.

   IEEE version: With a pilot recording (not one of the clips of Section~\ref{sec:livegap}) as camera, tracking ran at \V{t.hudon.fps} frames per second with the hand skeleton drawn and \V{t.hudoff.fps} without it, and a word appeared \V{t.hudon.e2e1} ms (p50) after the segmenter ended the sign.

14. Full version: Without a working GPU delegate the hand landmarker takes \V{t.cpu.hand} ms per frame and tracking drops to \V{t.cpu.fps} fps, below our \V{budget.fps} fps target.

   IEEE version: Without a working GPU delegate, tracking drops to \V{t.cpu.fps} fps, below our \V{budget.fps} fps target.

15. Full version: \emph{Translate}, despite its name, turns a sequence of isolated signs into a row of glosses; below the calibrated threshold it shows ``not sure'' with the top three candidates and adds nothing until the user picks one.

   IEEE version: \emph{Translate}, despite its name, turns a sequence of isolated signs into a row of glosses and shows ``not sure'' with the top three candidates below the calibrated threshold.

16. Full version: \emph{Practice} scores the next segment against a target sign as correct (top-1), close (top-3) or wrong.

   IEEE version: \emph{Practice} scores the next segment against a target sign, and \emph{Teach} keeps a few examples of a sign in the browser's storage (Section~\ref{sec:teach}).

17. Full version: On request, a serverless proxy sends only the recognized glosses to a hosted language model (\cfg{llama-3.1-8b-instant} on Groq), instructed to write one short sentence without adding facts, and the browser speaks it.

   IEEE version: On request, a serverless proxy sends only the recognized glosses to a hosted language model that writes one short sentence.

### `paper/ieee/sections/limitations.tex`

18. Full version: The author is not a fluent signer; the signs in the live recordings were learned from online ASL video dictionaries.

   IEEE version: The author is not a fluent signer; the signs in the live recordings were learned from online ASL video dictionaries, and how the \V{lg.signs} signs were chosen was not recorded.

19. Full version: Two controls would locate the remaining gap: running the live pipeline on test-split landmark sequences, which measures what the live code costs in distribution, and an oracle segmentation of the live clips, which separates segmentation from recognition.

   IEEE version: Two controls would locate the remaining gap: running the live pipeline on test-split landmark sequences, and an oracle segmentation of the live clips.

20. Full version: The mapping between the two vocabularies is AI-assisted and unconfirmed by a signer.

   IEEE version: The ASL Citizen label mapping is AI-assisted and unconfirmed by a signer, and v3's in-domain accuracy does not validate it.

21. Full version: The live clips are not an independent test of v3 either: they were analyzed before v3's webcam augmentation was designed, that augmentation targets the discrepancies they revealed, and Teach Dex reuses them.

   IEEE version: The live clips are not an independent test of v3 either: they were analyzed before v3's webcam augmentation was designed.

22. Full version: It does not handle continuous signing, fingerspelling, or signs outside the vocabulary, which it maps to the nearest known sign unless the confidence threshold rejects them.

   IEEE version: The model recognizes \V{data.signs} isolated signs; it does not handle continuous signing, fingerspelling, or signs outside the vocabulary, and it is not a translator.

23. Full version: The author is based in India, where Indian Sign Language is the relevant language; nothing here transfers to it without new data, such as INCLUDE \citep{sridhar2020include}.

   IEEE version: The author is based in India, where Indian Sign Language is the relevant language; nothing here transfers to it without new data.

24. Full version: Sign language technology should be built with, and ideally led by, Deaf communities \citep{bragg2019interdisciplinary, yin2021including,fox2023bestpractices}, and Deaf researchers have criticized how often the field sets its agenda without them \citep{desai2024systemic}.

   IEEE version: Sign language technology should be built with, and ideally led by, Deaf communities \cite{bragg2019interdisciplinary,yin2021including,fox2023bestpractices}, and Deaf researchers have criticized how often the field sets its agenda without them \cite{desai2024systemic}.

25. Full version: At the live accuracy we measured, however, the Practice mode would tell a learner that many correctly produced signs are wrong, and for some signs it fails every time; it could discourage learners or teach them to distort signs to please the model.

   IEEE version: At the live accuracy we measured, the app's Practice mode, which scores a sign against a target, would tell a learner that many correctly produced signs are wrong; it should be evaluated with learners and Deaf teachers before anyone relies on it.

26. Full version: The live clips show the author of this paper, who recorded them and consents to their use here; they are not released, and every figure in the paper shows landmarks only, never camera pixels.

   IEEE version: The live clips show the author, who consents to their use here; they are not released, and no ASL Citizen data or model trained on it is released either.

27. Full version: When the user asks for a sentence, the recognized glosses, and nothing else, are sent to a third-party language model provider, whose retention policy we have not audited; a gloss sequence can reveal what a user said.

   IEEE version: When the user asks the app for a sentence, only the recognized glosses are sent to a third-party language model provider.

28. Full version: Code, configurations and every result file read by the paper's generators are in the project repository; \cfg{paper/README.md} lists the commands.

   IEEE version: The code, configurations and result files behind every number are in the repository of the extended version \cite{shukla2026manuvox}.

### `paper/ieee/sections/livegap.tex`

29. Full version: We audited six candidate train--serve discrepancies, fixed each so that the browser reproduces Python (Figure~\ref{fig:pipeline}), and measured each fix on new recordings.

   IEEE version: We audited six candidate train--serve discrepancies, fixed each so that the browser reproduces Python\ifwithpipeline\ (Fig.~\ref{fig:pipeline})\fi, and measured each fix on new recordings.

30. Full version: Holistic, which produced the training data, crops each hand from the pose, so its \emph{left hand} block is the hand at the signer's left wrist.

   IEEE version: \emph{(a) Hand slots.} Holistic, which produced the training data, crops each hand from the pose, so its left-hand block is the hand at the signer's left wrist.

31. Full version: The shipped build already detected on the unmirrored frame.

   IEEE version: \emph{(b) Mirroring.} The shipped build already detected on the unmirrored frame.

32. Full version: Normalized coordinates give a body different proportions in a portrait phone recording and a wide webcam.

   IEEE version: \emph{(c) Aspect ratio.} Normalized coordinates give a body different proportions in a portrait phone recording and a wide webcam.

33. Full version: Training resampled by frame index (the dataset has no timestamps); the browser's frame rate varies, so the fix resamples on the capture-time clock.

   IEEE version: \emph{(d) Frame rate.} Training resampled by frame index (the dataset has no timestamps); the browser's frame rate varies, so the fix resamples on the capture-time clock.

34. Full version: The shipped build segmented by hand presence: a segment starts after \V{pres.start} frames with a hand and ends after \V{pres.end} without one or after \V{pres.max} s.

   IEEE version: \emph{(e) Segmentation.} The shipped build segmented by hand presence: a segment starts after \V{pres.start} frames with a hand and ends after \V{pres.end} without one or after \V{pres.max} s.

35. Full version: The fix is a state machine on motion energy, the speed of the wrist and fingertips on aspect-corrected coordinates, so (c) and (e) interact; its thresholds come from the validation sequences under an assumed frame rate (Appendix~\ref{app:livegap}).

   IEEE version: The fix is a state machine on motion energy, the speed of the wrist and fingertips on aspect-corrected coordinates, with thresholds from the validation sequences.

36. Full version: The shipped build showed the raw top-1 above a threshold set for \V{cal.targetprec}\% precision on validation.

   IEEE version: \emph{(f) Calibration and abstention.} The shipped build showed the raw top-1 above a threshold set for \V{cal.targetprec}\% precision on validation.

37. Full version: The fix divides the logits by the validation temperature and shows a word only above a validation threshold that maximizes the net score, in which a correct accepted word gains one, a wrong one costs one and ``not sure'' scores zero \citep{guo2017calibration,geifman2017selective}.

   IEEE version: The fix divides the logits by the validation temperature and shows a word only above a validation threshold that maximizes the net score, in which a correct accepted word gains one, a wrong one costs one and ``not sure'' scores zero.

38. Full version: The author recorded \V{lg.clips} clips with a laptop webcam: \V{lg.signs} everyday signs from the vocabulary (Table~\ref{tab:persign}), learned from online ASL video dictionaries, two takes each (median \V{lg.duration}\,s).

   IEEE version: The author recorded \V{lg.clips} clips with a laptop webcam: \V{lg.signs} everyday signs from the vocabulary, learned from online ASL video dictionaries, two takes each (median \V{lg.duration}\,s), unmirrored at \V{lg.width}$\times$\V{lg.height} and about \V{lg.fps} frames per second.

39. Full version: The author is both the developer and the only signer, and a pilot with the same person motivated the order of the fixes (Appendix~\ref{app:pilot}).

   IEEE version: The author is both the developer and the only signer, and a pilot with the same person motivated the order of the fixes.

40. Full version: One browser per clip runs every configuration in a seeded random order, three times each (Table~\ref{tab:livegapci}), and a clip's score is its mean over runs, so a browser clip can change by a third.

   IEEE version: One browser per clip runs every configuration in a seeded random order, three times each, and a clip's score is its mean over runs, so a browser clip can change by a third.

41. Full version: The baselines are the \emph{shipped build} itself (inference on the page's main thread, \V{lg.web.c0headbuild.fps} frames per second) and \emph{as shipped}, the current app with the shipped build's settings (\V{lg.web.c0asshipped.fps}).

   IEEE version: The baselines are the \emph{shipped build} itself (inference on the page's main thread, \V{lg.web.c0headbuild.fps} frames per second) and \emph{as shipped}, the current app with the shipped build's settings.

42. Full version: Table~\ref{tab:livegap} gives the ablation (plotted in Figure~\ref{fig:livegap}).

   IEEE version: Table~\ref{tab:livegap} gives the ablation.

43. Full version: With the hybrid and all fixes, clip top-1 reaches \V{lg.web.c6final.top1}\% \V{lg.web.c6final.top1ci} in the browser and \V{lg.py.c6final.top1}\% \V{lg.py.c6final.top1ci} in Python (top-3 \V{lg.web.c6final.top3}\% and \V{lg.py.c6final.top3}\%).

   IEEE version: With the hybrid and all fixes, clip top-1 reaches \V{lg.web.c6final.top1}\% \V{lg.web.c6final.top1ci} in the browser and \V{lg.py.c6final.top1}\% \V{lg.py.c6final.top1ci} in Python.

44. Full version: The test split also excludes sequences with too few hand frames, while a live clip without a segment counts as wrong, so the drop from test accuracy somewhat overstates the live gap.

   IEEE version: The test split excludes sequences with too few hand frames while a clip without a segment counts as wrong, so this drop somewhat overstates the live gap.

45. Full version: None of these top-1 changes is significant after correction at either unit, so we treat them as inconclusive.

   IEEE version: None of these top-1 changes is significant after correction at either unit.

46. Full version: The motion segmenter loses mostly whole signs it never segmented (\V{lg.py.c6final.noseg} clips in Python and \V{lg.web.c6final.noseg} per browser run without a classified segment, against \V{lg.py.xpresence.noseg} and \V{lg.web.xpresence.noseg}), and its segments are shorter (median \V{lg.web.c5segmenter.segms} against \V{lg.web.c4time.segms} ms in the browser).

   IEEE version: The motion segmenter loses mostly whole signs it never segmented (\V{lg.py.c6final.noseg} clips in Python and \V{lg.web.c6final.noseg} per browser run without a classified segment, against \V{lg.py.xpresence.noseg} and \V{lg.web.xpresence.noseg}).

47. Full version: Swapping in the hybrid before any fix gained \V{lg.web.c1model.dtop1} points in the browser (\V{lg.web.c1model.dtop1better} clips better, none worse; \V{lg.web.c1model.dtop1sig} at clip level, not at sign level, Holm $p$ = \V{lg.web.c1model.dtop1psign}) and \V{lg.py.c1model.dtop1} in Python, from a floor near zero.

   IEEE version: Swapping in the hybrid before any fix gained \V{lg.web.c1model.dtop1} points in the browser and \V{lg.py.c1model.dtop1} in Python, from a floor near zero.

48. Full version: Resampling by time (d), which cannot matter in Python, changed browser clip top-1 by \V{lg.web.c4time.dtop1} points, and calibration (f), which cannot change a segment's top-1, by \V{lg.web.c6final.dtop1}.

   IEEE version: Resampling by time (d), which cannot matter in Python, changed browser clip top-1 by \V{lg.web.c4time.dtop1} points, and calibration (f) by \V{lg.web.c6final.dtop1}.

49. Full version: Browser minus Python is \V{lg.pd.c6final.diff} points \V{lg.pd.c6final.ci} for the final pipeline and \V{lg.pd.xpresence.diff} \V{lg.pd.xpresence.ci} with presence segmentation (signs as the unit).

   IEEE version: Browser minus Python is \V{lg.pd.c6final.diff} points \V{lg.pd.c6final.ci} for the final pipeline (signs as the unit).

50. Full version: We see no evidence that the training tracker does better on these clips, but the intervals allow a sizeable advantage for Holistic, and the pipelines also differ in real-time frame dropping.

   IEEE version: We see no evidence that the training tracker does better on these clips, but the interval allows a sizeable advantage for Holistic, and the pipelines also differ in real-time frame dropping.

51. Full version: In both pipelines, \V{lg.failboth} of the \V{lg.signs} signs fail in every take, and for \V{lg.sharedconf} the most frequent wrong guess is the same in both (\V{lg.sharedconflist}; Table~\ref{tab:persign}).

   IEEE version: In both pipelines, \V{lg.failboth} of the \V{lg.signs} signs fail in every take.

52. Full version: Live, the calibration does not hold: the expected calibration error is \V{lg.cal.web.ece} in the browser (\V{cal.test.ece1} on test), and the threshold accepts \V{lg.cal.web.cov}\% of segments at \V{lg.cal.web.sel}\% selective accuracy (net score \V{lg.cal.web.net}, against \V{cal.test.net} on test), holding back \V{lg.cal.web.rejwrong}\% of the wrong guesses and \V{lg.cal.web.rejright}\% of the correct ones.

   IEEE version: Live, the calibration does not hold: the expected calibration error is \V{lg.cal.web.ece} in the browser (\V{cal.test.ece1} on test), and the threshold accepts \V{lg.cal.web.cov}\% of segments at \V{lg.cal.web.sel}\% selective accuracy.

53. Full version: Through the final pipeline, v3 reached clip top-1 \V{v3c.web.b}\% \V{v3c.web.bci} in the browser and \V{v3c.py.b}\% \V{v3c.py.bci} in Python, against \V{v3c.web.a}\% and \V{v3c.py.a}\% for v2 in the same runs: \V{v3c.web.diff} points in the browser (\V{v3c.web.better} clips better, \V{v3c.web.worse} worse; Holm $p$ = \V{v3c.web.p}, \V{v3c.web.sig}) and \V{v3c.py.diff} in Python (\V{v3c.py.better} better, \V{v3c.py.worse} worse; $p$ = \V{v3c.py.p}, \V{v3c.py.sig}; Table~\ref{tab:v3}).

   IEEE version: Through the final pipeline, v3 reached clip top-1 \V{v3c.web.b}\% \V{v3c.web.bci} in the browser and \V{v3c.py.b}\% \V{v3c.py.bci} in Python, against \V{v3c.web.a}\% and \V{v3c.py.a}\% for v2 in the same runs: \V{v3c.web.diff} points in the browser (Holm $p$ = \V{v3c.web.p}, \V{v3c.web.sig}) and \V{v3c.py.diff} in Python ($p$ = \V{v3c.py.p}, \V{v3c.py.sig}).

54. Full version: Teach Dex, our personalization feature, borrows the idea without retraining: for the signs a user has taught, it adds the cosine similarity between a segment's embedding and the user's examples to the frozen model's calibrated logits, with a weight fitted on validation signers rather than on the user's clips.

   IEEE version: For each sign a user has taught, Teach Dex takes the highest cosine similarity $s_c$ between a segment's embedding (the model's \V{td.embdim}-value penultimate layer) and the user's examples of sign $c$, and adds $w\,(s_c - b)$ to that sign's calibrated logit; the weight $w$ (\V{td.v2.weight}) and offset $b$ (\V{td.v2.offset}) were fitted on validation signers, not on the user's clips.

55. Full version: Through the app's own Teach panel, the take-1 clip of each sign was replayed to supply up to \V{td.examples} examples, each after the panel's countdown (a still lead-in of \V{td.leadin}\,s kept the countdown off the sign), so a sign's examples repeat one production, and \V{td.v2.noexmin}--\V{td.v2.noexmax} signs per run got none.

   IEEE version: In a separate record-then-test evaluation, take 1 of each sign supplied up to \V{td.examples} examples, all from one production, and take 2 was recognized with them in \V{td.v2.runs} browser runs.

56. Full version: With v2, clip top-1 on take 2 rose from \V{td.v2.model}\% \V{td.v2.modelci} to \V{td.v2.teach}\% \V{td.v2.teachci}, \V{td.v2.diff} points \V{td.v2.ci} (\V{td.v2.better} clips better, \V{td.v2.worse} worse; Holm $p$ = \V{td.v2.p}, \V{td.v2.sig}).

   IEEE version: With v2, clip top-1 on take 2 rose from \V{td.v2.model}\% \V{td.v2.modelci} to \V{td.v2.teach}\% \V{td.v2.teachci}, \V{td.v2.diff} points \V{td.v2.ci} (Holm $p$ = \V{td.v2.p}, \V{td.v2.sig}).

57. Full version: A v3 run of the same test was compromised by its harness.

   IEEE version: A v3 run of the same test was compromised by its harness; the extended version reports it.

### `paper/ieee/sections/models.tex`

58. Full version: The \emph{GRU} projects each frame and runs a two-layer bidirectional GRU \citep{cho2014gru}; the \emph{CNN} is a stack of residual depthwise-separable temporal convolutions \citep{howard2017mobilenets,bai2018tcn}; the \emph{Transformer} is a four-layer encoder \citep{vaswani2017attention}.

   IEEE version: The \emph{GRU} runs a two-layer bidirectional GRU \cite{cho2014gru}; the \emph{CNN} is a stack of residual depthwise-separable temporal convolutions \cite{howard2017mobilenets,bai2018tcn}; the \emph{Transformer} is a four-layer encoder \cite{vaswani2017attention}.

59. Full version: The \emph{v1 recipe} trains for up to \V{hp.v1.epochs} epochs with AdamW, cosine decay, label smoothing and geometric and temporal augmentation, with early stopping on validation top-1.

   IEEE version: The \emph{v1 recipe} trains for up to \V{hp.v1.epochs} epochs with AdamW, cosine decay, label smoothing and geometric and temporal augmentation.

60. Full version: The \emph{v2 recipe} trains up to \V{hp.v2.epochs} epochs at a lower learning rate, with an exponential moving average of the weights (decay \V{hp.v2.ema}) \citep{polyak1992averaging}, frame dropout, time and landmark masking and hand rotation.

   IEEE version: The \emph{v2 recipe} trains up to \V{hp.v2.epochs} epochs at a lower learning rate, with an exponential moving average of the weights \cite{polyak1992averaging}, frame dropout, time and landmark masking and hand rotation.

61. Full version: Each configuration has \V{hp.seeds} seeds (Table~\ref{tab:hparams}); a hybrid run took about \V{m.d.trainmin} minutes on a Kaggle \V{m.gpu}.

   IEEE version: Each configuration has \V{hp.seeds} seeds; a hybrid run took about \V{m.d.trainmin} minutes on a Kaggle \V{m.gpu}.

62. Full version: Among the v1 models, the transformer's difference from the CNN is \V{cmp.tf.sig} and the GRU differs from it by \V{cmp.gru.diff} points (Table~\ref{tab:comparisons}).

   IEEE version: In paired tests over the \V{cmp.n} test sequences with Holm correction\ifwithcomparisons\ (Table~\ref{tab:comparisons})\fi\ (differences computed before rounding), the transformer's difference from the CNN is \V{cmp.tf.sig} and the GRU differs from it by \V{cmp.gru.diff} points.

63. Full version: These tests treat the \V{cmp.n} test sequences as independent and average over seeds, so they exclude seed and signer variation.

   IEEE version: These tests treat the sequences as independent and average over seeds, so they exclude seed and signer variation.

64. Full version: Per signer, the hybrid's top-1 ranges from \V{sig.d.min}\% to \V{sig.d.max}\% (Table~\ref{tab:persigner}).

   IEEE version: Per signer, the hybrid's top-1 ranges from \V{sig.d.min}\% to \V{sig.d.max}\%, and it beats the CNN for all four test signers, which is weak evidence about new ones.

65. Full version: The shipping rule took the highest \emph{test} top-1 with a p95 browser latency under \V{budget.ms} ms and a file under ten megabytes (Figure~\ref{fig:acclat} in the appendix).

   IEEE version: The shipping rule took the highest \emph{test} top-1 with a p95 browser latency under \V{budget.ms} ms and a file under ten megabytes; the size cap alone excludes an ensemble of the hybrid's three seeds (\V{m.d.ens.top1}\% at \V{m.d.ens.mb}\,MB in fp16).

66. Full version: The fitted temperature is below one (\V{cal.T}): the raw model is underconfident, consistent with label smoothing (Appendix~\ref{app:calib}).

   IEEE version: The fitted temperature is below one (\V{cal.T}): the raw model is underconfident, consistent with label smoothing.

67. Full version: On the test split, temperature scaling lowers the expected calibration error (\V{cal.bins} equal-width bins) from \V{cal.test.ece0} \V{cal.test.ece0ci} to \V{cal.test.ece1} \V{cal.test.ece1ci} and the negative log-likelihood from \V{cal.test.nll0} to \V{cal.test.nll1}.

   IEEE version: On the test split, temperature scaling lowers the expected calibration error (\V{cal.bins} equal-width bins) from \V{cal.test.ece0} \V{cal.test.ece0ci} to \V{cal.test.ece1} \V{cal.test.ece1ci}.

68. Full version: The validation threshold (\V{cal.thr}\%) accepts \V{cal.test.cov}\% of test sequences at \V{cal.test.sel}\% selective accuracy, close to its validation operating point (\V{cal.val.cov}\% at \V{cal.val.sel}\%), so it transfers to unseen signers of the same dataset.

   IEEE version: The validation threshold (\V{cal.thr}\%) accepts \V{cal.test.cov}\% of test sequences at \V{cal.test.sel}\% selective accuracy, close to its validation operating point (\V{cal.val.cov}\% at \V{cal.val.sel}\%).

69. Full version: Each test set was evaluated once, after v3 was selected (Table~\ref{tab:v3}).

   IEEE version: Epoch and seed are selected by the unweighted mean of the two validation top-1 scores, and each test set was evaluated once, after v3 was selected\ifwithvthree\ (Table~\ref{tab:v3})\fi.

70. Full version: On ASL Citizen, v2 reaches \V{cit.v2.top1}\% (95\% bootstrap interval over signers \V{cit.v2.top1cis}), \V{cit.v2.drop} points below its own test split, and less on signs mapped through a variant (\V{cit.v2.variant}\%) than on exact matches (\V{cit.v2.exact}\%), which may reflect the mapping or different sign forms.

   IEEE version: On ASL Citizen, v2 reaches \V{cit.v2.top1}\% (95\% bootstrap interval over signers \V{cit.v2.top1cis}), \V{cit.v2.drop} points below its own test split, less on signs mapped through a variant (\V{cit.v2.variant}\%) than on exact matches (\V{cit.v2.exact}\%), and its calibration degrades: the error after scaling is \V{cit.v2.ece}, and the threshold accepts only \V{cit.v2.cov}\% of videos.

71. Full version: On the Google ISLR test split, v3 reaches \V{v3c.ks.b}\% against \V{v3c.ks.a}\% for the deployed v2 file, \V{v3c.ks.diff} points, positive for each test signer (\V{v3c.ks.cmin} to \V{v3c.ks.cmax}).

   IEEE version: On the Google ISLR test split, v3 reaches \V{v3c.ks.b}\% against \V{v3c.ks.a}\% for the deployed v2 file, \V{v3c.ks.diff} points\ifwithvthree\else\ \V{v3c.ks.ci} (Holm $p$ \V{v3c.ks.p})\fi, positive for each test signer.

72. Full version: This compares two files, a validation-selected v3 seed and v2's seed 0, not two training procedures, and v3 changes both data and augmentation: together they lost nothing on the first dataset and gained a little.

   IEEE version: This compares two files, a validation-selected v3 seed and v2's seed 0, not two training procedures.

### `paper/ieee/sections/related.tex`

73. Full version: Pose-based models classify body and hand keypoints instead of pixels, which removes appearance and makes on-device inference cheap: SPOTER uses a small transformer \citep{bohacek2022spoter}, and OpenHands releases such models for several sign languages \citep{selvaraj2022openhands}.

   IEEE version: Pose-based models classify body and hand keypoints instead of pixels, which removes appearance and makes on-device inference cheap \cite{bohacek2022spoter,selvaraj2022openhands}.

74. Full version: \citet{moryossef2021pose} show with two pose estimators what the skeletal representation loses, so the landmarks available at serving time limit what a model can recognize.

   IEEE version: Moryossef et al.~\cite{moryossef2021pose} show with two pose estimators what the skeletal representation loses, so the landmarks available at serving time limit what a model can recognize.

75. Full version: The Google ISLR competition \citep{kaggle2023islr} released MediaPipe Holistic landmarks rather than video, collected with the partners behind the PopSign game \citep{starner2023popsign}.

   IEEE version: The Google ISLR competition \cite{kaggle2023islr} released MediaPipe Holistic landmarks rather than video.

76. Full version: Selective prediction abstains below a threshold \citep{geifman2017selective}, as our ``not sure'' does.

   IEEE version: Temperature scaling corrects miscalibration \cite{guo2017calibration}, selective prediction abstains below a threshold \cite{geifman2017selective}, as our ``not sure'' does, and calibration degrades under dataset shift \cite{ovadia2019shift}.

77. Full version: Each step reads its constants from \cfg{features\_spec.json} or the model metadata, which the Python training code also reads.}

   IEEE version: Teach Dex, our personalization feature, borrows the idea of prototypical networks \cite{snell2017prototypical} without retraining.
