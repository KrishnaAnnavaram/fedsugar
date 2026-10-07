# The writing standard: ASD-STE100 Simplified Technical English for fedsugar

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

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

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **survey file** | The BRFSS 2015 CSV file with 21 indicators and the target. | dataset (alone), database |
| **record** | One row of the survey file: one respondent. | sample (for a row), patient, entry |
| **indicator** | One of the 21 input columns, for example `HighBP`. | variable, attribute |
| **target** | The column `Diabetes_012`: 0, 1 or 2. | label column, outcome (in prose) |
| **class** | One target value: `no_diabetes`, `prediabetes` or `diabetes`. | category, label (for a value) |
| **split** | The stratified train, validation and test parts of the records. | fold, partition (for train and test) |
| **client** | A simulated data holder in federated training. It keeps its records. | site, node, party, silo (for a holder) |
| **partition** | The assignment of training records to clients: `iid`, `dirichlet` or `silo`. | split (for clients), shard |
| **alpha** | The concentration value of the Dirichlet partition. A small alpha gives strong label skew. | skew level |
| **label skew** | The mean total-variation distance between the class mix of each client and the global mix. | heterogeneity, non-IID degree |
| **server** | The component that makes the initial model and aggregates the updates. | aggregator, coordinator |
| **model** | The softmax regression (`logreg`) or the one-layer network (`mlp`). | classifier (in prose), network (for `logreg`) |
| **weights** | The parameter arrays of a model. | parameters (in prose), coefficients |
| **update** | The difference between the weights of a client after local training and the global weights. | gradient (for an update), delta (in prose) |
| **round** | One cycle: send weights, train on clients, aggregate. | iteration, epoch (for a round) |
| **local epoch** | One pass of a client over its own records in one round. | step, local round |
| **strategy** | The aggregation rule: `fedavg`, `fedprox` or `fedadam`. | algorithm, optimizer (for the server rule) |
| **class weights** | The weight of each class in the loss: `global`, `local` or `none`. | sample weights (in prose), balancing |
| **centralized model** | The same model trained on all training records at one location. | pooled model, baseline (alone) |
| **federated model** | The model that the server aggregates over the rounds. | global model (in prose), FL model |
| **local-only baseline** | The mean result of models that each client trains alone. | isolated training |
| **macro-F1** | The mean of the F1 values of the three classes. The main metric. | F1 (alone), accuracy |
| **parallel time** | The sum over rounds of the slowest client time plus the aggregation time. A value from measured times. | estimated time, simulated speed-up |
| **sequential time** | The sum of all client times and aggregation times, as measured in the simulation. | total time (alone) |
| **communication** | The megabytes that clients and the server send, at 4 bytes for each parameter. | traffic, bandwidth |
| **grid** | The set of runs in a TOML file: seeds × partitions × strategies × models. | sweep file, experiment plan |
| **synthetic data** | The generated rows of `fedsugar synth`. They are not survey records. | fake data, mock data, sample data |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read the survey file and check it against the column contract. |
| **scale** | Change each indicator to the range 0 to 1 with its documented limits. |
| **partition** | Assign the training records to clients. |
| **train** | Run mini-batch SGD on records with the class-weighted loss. |
| **aggregate** | Make new global weights from the client updates with the strategy. |
| **select** | Choose the round or the epoch with the best validation macro-F1. |
| **clip** | Make the norm of an update not larger than the clip value. |
| **measure** | Get a time with `time.perf_counter`. |
