# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of cellsight and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the cellsight documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **image** | One image file of one cell. | sample, picture, photo |
| **class** | One of the five cell types: `basophil`, `erythroblast`, `monocyte`, `myeloblast`, `seg_neutrophil`. | category, label (for the type) |
| **critical class** | `myeloblast`, the leukemia-relevant class. Its recall is reported on its own. | positive class |
| **manifest** | The index of all images: path, class, group and SHA-256. | catalog, file list |
| **near-duplicate** | An image whose perceptual hash is within the hash distance of another image, after rotation or flip. | copy, clone |
| **duplicate group** | A set of images that near-duplicate links connect. | cluster |
| **split group** | The unit that a split keeps on one side: the slide group merged with the duplicate groups. | fold group, block |
| **development images** | All images outside the test set. Folds, tuning and stacking use only these. | training set (for all of them) |
| **test set** | The images that the code predicts once, at the end. | holdout, evaluation set |
| **base model** | One classifier inside the ensemble: a light model or a backbone. | learner, member |
| **light model** | A scikit-learn base model on handcrafted features (`hist_logreg`, `shape_rf`, `thumb_knn`). | baseline model |
| **backbone** | A pretrained timm network (7 are configured). | architecture, net |
| **OOF probabilities** | Out-of-fold probabilities: predictions for development images from models that did not see them. | validation predictions |
| **stacker** | The logistic-regression meta-model on the OOF probabilities. | meta-learner, blender |
| **bundle** | The saved final model: base models and the stacker. | checkpoint (for the bundle) |
| **occlusion map** | The drop of the class log-probability when a patch hides each region. | saliency, heat map (alone) |
| **focus ratio** | The share of positive evidence in the cell mask divided by the share of area in the mask. | attention score |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **index** | Scan the image folder and make the manifest. |
| **dedupe** | Find near-duplicates and assign the split groups. |
| **split** | Divide images by split group, stratified by class. |
| **stack** | Fit the stacker on the OOF probabilities. |
| **evaluate** | Run dedupe, split, stacking and the single test pass, then write the report. |
| **fine-tune** | Train a backbone in the frozen stage and then in the full stage. |
| **explain** | Make an occlusion map and a focus ratio for one image. |
