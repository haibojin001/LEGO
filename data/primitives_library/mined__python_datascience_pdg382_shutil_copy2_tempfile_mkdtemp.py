# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg382::shutil.copy2+tempfile.mkdtemp
# name: shutil_tempfile_primitive
# summary: Uses shutil.copy2, tempfile.mkdtemp across 2 repos
# anchor_symbols: ['shutil.copy2', 'tempfile.mkdtemp']
# observed in 2 repos: ['okfn-brasil__serenata-de-amor', 'ruc-datalab__DeepAnalyze']...

# --- from okfn-brasil__serenata-de-amor::rosie/rosie/chamber_of_deputies/tests/test_adapter.py::TestAdapter.setUpClass ---
def setUpClass(cls):
        cls.temp_path = mkdtemp()
        to_copy = (
            'companies.xz',
            'reimbursements-2009.csv',
            'reimbursements-2010.csv',
            'reimbursements-2011.csv',
            'reimbursements-2012.csv',
            'reimbursements-2016.csv'
        )
        for source in to_copy:
            target = source
            if source == 'companies.xz':
                target = Adapter.COMPANIES_DATASET
            shutil.copy2(FIXTURES / source, Path(cls.temp_path) / target)

# --- from ruc-datalab__DeepAnalyze::example/4c_competition/quick_start.py::process_streaming_chat ---
def process_streaming_chat(uploaded_files, user_instruction, api_key):
    """Run streaming chat analysis."""
    global client
    
    # Initialize client
    client = openai.OpenAI(
        base_url=DEFAULT_API_BASE,
        api_key=api_key,
    )
    
    print("🔄 Starting analysis...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    files_to_upload = []
    file_objects = []
    supported_extensions = get_supported_file_extensions()
    
    try:
        # Handle uploaded files
        if uploaded_files:
            for file_path in uploaded_files:
                if not os.path.exists(file_path):
                    continue
                
                file_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_name)[1].lower()
                
                # Check for ZIP
                if file_ext == '.zip':
                    extract_dir = os.path.join(temp_dir, f"extracted_{os.path.splitext(file_name)[0]}")
                    os.makedirs(extract_dir, exist_ok=True)
                    extracted_files = extract_zip_file(file_path, extract_dir)
                    
                    if extracted_files:
                        for extracted_file in extracted_files:
                            extracted_name = os.path.basename(extracted_file)
                            extracted_ext = os.path.splitext(extracted_name)[1].lower()
                            
                            if extracted_ext in supported_extensions:
                                dest_path = os.path.join(temp_dir, extracted_name)
                                counter = 1
                                while os.path.exists(dest_path):
                                    name, ext = os.path.splitext(extracted_name)
                                    dest_path = os.path.join(temp_dir, f"{name}_{counter}{ext}")
                                    counter += 1
                                
                                shutil.copy2(extracted_file, dest_path)
                                files_to_upload.append(dest_path)
                else:
                    if file_ext in supported_extensions:
                        dest_path = os.path.join(temp_dir, file_name)
                        shutil.copy2(file_path, dest_path)
                        files_to_upload.append(dest_path)
            
            # Upload files to API
            for file_path in files_to_upload:
                try:
                    with open(file_path, "rb") as f:
                        file_obj = client.files.create(file=f, purpose="file-extract")
                        file_objects.append(file_obj)
                except:
                    pass
        
        file_names = [os.path.basename(path) for path in files_to_upload]
        
        # Use provided or default instruction
        if not user_instruction.strip():
            if files_to_upload:
                user_instruction = (
                    f"Please analyze the following data files {', '.join(file_names)}, "
                    "perform EDA, and generate visualizations. Focus on relationships, trends, and key insights."
                )
            else:
                user_instruction = "Please conduct a conversational analysis and provide detailed insights."
        
        print("\n" + "=" * 60)
        
        # Build messages
        if files_to_upload:
            messages = [
                {
                    "role": "user",
                    "content": user_instruction,
                    "file_ids": [file_obj.id for file_obj in file_objects],
                }
            ]
        else:
            messages = [{"role": "user", "content": user_instruction}]
        
        # Pass api_key via extra_body
        extra_body = {"api_key": api_key} if api_key else {}
        
        # Create streaming request
        try:
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                stream=True,
                extra_body=extra_body,
            )
        except openai.InternalServerError as e:
            raise Exception(f"❌ API server error: {e}")
        except openai.APIError as e:
            raise Exception(f"❌ API error: {e}")
        except Exception as e:
            raise Exception(f"❌ Connection error: {e}")
        
        full_response = ""
        collected_files = []
        downloadable_files = []
        
        # Stream output
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
            
            if hasattr(chunk, "generated_files") and chunk.generated_files:
                collected_files.extend(chunk.generated_files)
        
        print("\n" + "=" * 60)
        
        # Download generated files
        if collected_files:
            for file_info in collected_files:
                filename = file_info.get("name", f"generated_{len(downloadable_files)}.txt")
                url = file_info.get("url", "")
                if url:
                    local_path = download_file_from_url(url, filename, temp_dir)
                    if local_path:
                        downloadable_files.append(local_path)
        
        print(f"\n✅ Analysis complete (generated files: {len(collected_files)})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup temp files (optional)
        # if temp_dir and os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        pass
