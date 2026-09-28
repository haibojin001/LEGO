# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg113::logging.debug
# name: logging_primitive
# summary: Uses logging.debug across 9 repos
# anchor_symbols: ['logging.debug']
# observed in 9 repos: ['ACEnglish__truvari', 'HoloClean__holoclean', 'awslabs__gluonts', 'cleanlab__cleanlab', 'makcedward__nlp']...

# --- from ploomber__ploomber::tests/dag/test_dagconfigurator.py::touch_root ---
def touch_root(product):
    logging.debug("This should not appear...")
    logging.info("Logging...")
    Path(str(product)).touch()

# --- from HoloClean__holoclean::holoclean.py::Session.setup_domain ---
def setup_domain(self):
        status, domain_time = self.domain_engine.setup()
        logging.info(status)
        logging.debug('Time to setup the domain: %.2f secs', domain_time)

# --- from HoloClean__holoclean::holoclean.py::Session.detect_errors ---
def detect_errors(self, detect_list):
        status, detect_time = self.detect_engine.detect_errors(detect_list)
        logging.info(status)
        logging.debug('Time to detect errors: %.2f secs', detect_time)

# --- from makcedward__nlp::aion/embeddings/skip_thoughts.py::SkipThoughtsEmbeddingsTorch.encode ---
def encode(self, sentences, output_format='torch'):
        transformed_sentences = self.process(sentences)
        
        algo = self.get_algorithm(self.vocabs)
        inputs = Variable(LongTensor(transformed_sentences))
        outpus = algo(inputs, lengths=[len(words) for words in transformed_sentences])
        
        if output_format == 'np':
            return self.to_numpy_layer(outpus)
        elif output_format == 'torch':
            return outpus

# --- from modin-project__modin::stress_tests/test_kaggle_ipynb.py::test_kaggle6 ---
def test_kaggle6(generate_dataset):
    columns = []
    dtypes = []
    generate_dataset("test.csv", columns, dtypes)
    generate_dataset("train.csv", columns, dtypes)

    ipynb = subprocess.Popen(
        ["python", "kaggle6.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=KAGGLE_DIR_PATH,
    )
    outs, errs = ipynb.communicate()

    if ipynb.returncode:
        logging.debug("Error message\n-------------\n %s", errs.decode("utf-8"))

    logging.info("Finished kaggle6")
    assert ipynb.returncode == 0

# --- from ACEnglish__truvari::truvari/region_vcf_iter.py::merge_region_tree_overlaps ---
def merge_region_tree_overlaps(tree):
    """
    Runs IntervalTree.merge_overlaps on all trees. Info Logs the count of chromosomes with overlapping regions 
    and Debug Logs the chromosome names with overlapping regions.
    """
    chr_with_overlaps = []
    for i in tree:
        pre_len = len(tree[i])
        tree[i].merge_overlaps()
        post_len = len(tree[i])
        if pre_len != post_len:
            chr_with_overlaps.append(i)
    if chr_with_overlaps:
        logging.info("Found %d chromosomes with overlapping regions",
                     len(chr_with_overlaps))
        logging.debug("CHRs: %s", chr_with_overlaps)

# --- from modin-project__modin::stress_tests/test_kaggle_ipynb.py::test_kaggle13 ---
def test_kaggle13(generate_dataset):
    columns = [
        "Id",
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
        "Species",
    ]
    dtypes = [int, float, float, float, float, str]
    generate_dataset("Iris.csv", columns, dtypes)

    ipynb = subprocess.Popen(
        ["python", "kaggle13.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=KAGGLE_DIR_PATH,
    )
    outs, errs = ipynb.communicate()

    if ipynb.returncode:
        logging.debug("Error message\n-------------\n %s", errs.decode("utf-8"))

    logging.info("Finished kaggle13")
    assert ipynb.returncode == 0

# --- from cleanlab__cleanlab::cleanlab/experimental/coteaching.py::evaluate ---
def evaluate(test_loader, model1, model2):
    print("Evaluating Co-Teaching Model")
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()  # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()

    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)
    return acc1, acc2

# --- from awslabs__gluonts::src/gluonts/nursery/few_shot_prediction/src/meta/datasets/cheat.py::CheatArtificialDataModule.evaluate_model ---
def evaluate_model(
        self,
        model: nn.Module,
        quantiles: List[str],
        test: bool = False,
    ) -> List[Dict[str, float]]:
        metrics = []

        self.setup()
        pred = []
        if test:
            split = self.splits.test()
            dl = self.test_dataloader()
        else:
            split = self.splits.val()
            dl = self.val_dataloader()

        for batch in tqdm(
            dl,
            desc=f"generating predictions for dataset {self.name()} with support set size",
        ):
            pred.append(model(supps=batch.support_set, query=batch.query_past))

        # redo standardization for evaluation
        pred = split.data().rescale_dataset(torch.cat(pred, dim=0))
        pred = tensor_to_np(pred)

        # use only the length that should be included for evaluation
        pred = pred[:, : self.prediction_length, ...]

        m = compute_metrics(
            pred,
            split.evaluation(),
            quantiles=quantiles,
            seasonality=get_seasonality(self.meta.freq),
        )
        metrics.append(m)
        return metrics

# --- from piskvorky__gensim::gensim/examples/dmlcz/gensim_xml.py::generateSimilar ---
def generateSimilar(corpus, index, method):
    for docNo, topSims in enumerate(index):  # for each document
        # store similarities to the following file
        outfile = os.path.join(corpus.articleDir(docNo), 'similar_%s.xml' % method)

        articles = []  # collect similars in this list
        for docNo2, score in topSims:  # for each most similar article
            if score > MIN_SCORE and docNo != docNo2:
                source, (intId, pathId) = corpus.documents[docNo2]
                meta = corpus.getMeta(docNo2)
                suffix, author, title = '', meta.get('author', ''), meta.get('title', '')
                articles.append(ARTICLE % locals())  # add the similar article to output
                if len(articles) >= MAX_SIMILAR:
                    break

        # now `articles` holds multiple strings in similar_*.xml format
        if SAVE_EMPTY or articles:
            output = ''.join(articles)  # concat all similars to one string
            if not DRY_RUN:  # only open output files for writing if DRY_RUN is false
                logging.info("generating %s (%i similars)", outfile, len(articles))
                outfile = open(outfile, 'w')
                outfile.write(SIMILAR % output)  # add xml headers and print to file
                outfile.close()
            else:
                logging.info("would be generating %s (%i similars):%s\n", outfile, len(articles), output)
        else:
            logging.debug("skipping %s (no similar found)", outfile)

# --- from makcedward__nlp::aion/embeddings/infersent_lib/train_nli.py::evaluate ---
def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc

# --- from ruc-datalab__DeepAnalyze::playground/TableQA/tests/wikitq.py::create_prompt_from_wikitq ---
def create_prompt_from_wikitq(
    item, is_think=False, table_format="markdown", perturbation="none"
):
    """
    Create prompt for WikiTableQuestion item with different table formats and perturbations

    Args:
        item: The input item containing table and question
        is_think: Whether to use thinking prompt template
        table_format: Format for table representation - "markdown", "csv", or "dataframe"
        perturbation: Type of table perturbation - "none", "row_shuffle", "col_shuffle", "both_shuffle"

    Returns:
        Formatted prompt string
    """
    table_str = ""
    if "table" in item:
        # Deep copy table data to avoid modifying original data
        import copy
        import random

        table = copy.deepcopy(item["table"])

        # Get headers and data rows
        header = table[0] if len(table) > 0 else []
        data_rows = table[1:] if len(table) > 1 else []

        # Apply perturbation
        row_shuffle = perturbation in ["row_shuffle", "both_shuffle"]
        col_shuffle = perturbation in ["col_shuffle", "both_shuffle"]

        # Apply column perturbation
        if col_shuffle and len(header) > 1:
            # Generate new column order
            col_indices = list(range(len(header)))
            random.shuffle(col_indices)

            # Reorder headers and data rows
            header = [header[i] for i in col_indices]
            data_rows = [
                [row[i] if i < len(row) else "" for i in col_indices]
                for row in data_rows
            ]

        # Apply row perturbation
        if row_shuffle and len(data_rows) > 1:
            # Keep headers unchanged, randomly shuffle data rows
            random.shuffle(data_rows)

        # Recombine table
        shuffled_table = [header] + data_rows

        # Generate table string based on selected format
        if table_format == "markdown":
            # Create Markdown format table string representation
            for row in shuffled_table:
                table_str += " | ".join([str(cell) for cell in row]) + "\n"

        elif table_format == "csv":
            # Create CSV format table string representation
            for row in shuffled_table:
                table_str += ",".join([f'"{str(cell)}"' for cell in row]) + "\n"

        elif table_format == "dataframe":
            # Import pandas library
            import pandas as pd

            # Create DataFrame-like table representation
            if len(shuffled_table) > 0:
                # Get headers and data
                headers = shuffled_table[0]
                data = shuffled_table[1:] if len(shuffled_table) > 1 else []

                # Create pandas DataFrame
                df = pd.DataFrame(data, columns=headers)

                # Configure pandas display options to show complete table
                pd.set_option("display.expand_frame_repr", False)
                pd.set_option("display.width", 500000)
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.max_colwidth", None)

                # Convert DataFrame to string
                table_str = df.to_string(index=True)

                # Add DataFrame information line
                table_str += f"\n\n[{len(data)} rows x {len(headers)} columns]\n"

                pd.reset_option("display.expand_frame_repr")
                pd.reset_option("display.width")
                pd.reset_option("display.max_rows")
                pd.reset_option("display.max_columns")
                pd.reset_option("display.max_colwidth")

        else:
            # Default use Markdown format
            for row in shuffled_table:
                table_str += " | ".join([str(cell) for cell in row]) + "\n"

        # If perturbation was applied, add explanatory note
        if perturbation != "none" and not is_think:  # Add note in non-thinking mode
            notes = []
            if row_shuffle:
                notes.append("Row order has been randomly shuffled (except header)")
            if col_shuffle:
                notes.append("Column order has been randomly shuffled")

            if notes:
                table_str += "\nNote: " + ", ".join(notes) + "\n"

    # Use different templates based on thinking requirement
    if is_think:
        from prompt_think import COT_PROMPT_TEMPLATE

        prompt = COT_PROMPT_TEMPLATE.format(table=table_str, question=item["question"])
    else:
        from prompt import COT_PROMPT_TEMPLATE

        prompt = COT_PROMPT_TEMPLATE.format(table=table_str, question=item["question"])

    return prompt
