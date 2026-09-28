def update_predictions():
    '''Runs an iteration of model predictions on
    selected symbols and saves output in database
    '''
    db = DataBase([], DIR_DB)
    df_sym = get_df_sym_filter(db, LS_SEC, LS_IND)
    c_error = collections.Counter()
    ls_skip = []
    while 1:
        dt_error = {}
        idx_not_skip = -df_sym['sym'].isin(ls_skip)
        for i, tup in tqdm(df_sym[idx_not_skip].iterrows(), total=df_sym[idx_not_skip].shape[0]):
            while pause:
                time.sleep(BUFFER_SECONDS)
            sym = tup['sym']
            try:
                time.sleep(BUFFER_SECONDS)
                df_c = get_df_c(sym, DATE_STR, LIVE_DATA, db, TARGET_PROFIT, TARGET_LOSS)
                df_proba = get_df_proba(df_c, tup_model)
                if not df_proba.empty:
                    df_proba.to_sql('proba', db.conn, if_exists='append', index=0)
            except Exception as e:
                dt_error[sym] = ERROR_EXCEPTION.format(type(e).__name__, e) # traceback.print_exc()
                c_error.update([sym])
        if dt_error:
            num_runs = df_sym.shape[0]
            [print(ERROR_SUMMARY.format(sym, dt_error[sym])) for sym in dt_error]
            print(ERROR_PCT.format(len(dt_error), num_runs, len(dt_error)/num_runs))
        ls_skip =  [k for k, v in c_error.items() if v > ERROR_THRESHOLD] # skip symbols with too many errors
        print(MSG_SKIP.format(ls_skip))