# ##############################################################################
# File: async_api.py
# Author: Yinhong Liu
# Date: Feb. 6, 2024
# ##############################################################################

import os
import time
from openai import OpenAI
import openai
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import datetime


class Timer(object):
    def __init__(self):
        self.__start = time.time()

    def start(self):
        self.__start = time.time()

    def get_time(self, restart=True, format=False):
        end = time.time()
        span = end - self.__start
        if restart:
            self.__start = end
        if format:
            return self.format(span)
        else:
            return span

    def format(self, seconds):
        return datetime.timedelta(seconds=int(seconds))

    def print(self, name):
        print(name, self.get_time())


class OpenAIRequestManager:
    def __init__(self, response_extractor, api_params={}):
        # Global api parameters
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if 'engine' not in api_params:
            api_params['engine'] = 'gpt-3.5-turbo'
        if 'temperature' not in api_params:
            api_params['temperature'] = 0.2
        if 'max_tokens' not in api_params:
            api_params['max_tokens'] = 128
        if 'logprobs' not in api_params:
            api_params['logprobs'] = False
        if 'top_logprobs' not in api_params:
            api_params['top_logprobs'] = 5
        if 'attempt_num' not in api_params:
            api_params['attempt_num'] = 10        
        if 'buffer_path' not in api_params:
            api_params['buffer_path'] = './temp_buffer.jsonl'
        with open(api_params['buffer_path'], 'w') as f:
            pass
        self.response_extractor = response_extractor
        self.outbuf = open(api_params['buffer_path'], 'a')
        self.lock = threading.Lock()
        self.api_params = api_params
        self.client = OpenAI()

    def write_result(self, result):
        self.lock.acquire()
        self.outbuf.write(json.dumps(result, ensure_ascii=False) + '\n')
        self.outbuf.flush()
        self.lock.release()

    def openai_api_call(self, prompt):
        id, prompt = prompt
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        attempt = 0
        wait_sec = 0.1
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.api_params['engine'],
                    messages=messages,
                    temperature=self.api_params['temperature'],
                    max_tokens=self.api_params['max_tokens'],
                    logprobs=self.api_params['logprobs'],
                    top_logprobs=self.api_params['top_logprobs'] if self.api_params['logprobs'] else None,
                )
                result = self.response_extractor(response)
                result['id'] = id
                self.write_result(result)
                break
            except Exception as e:
                print(e)
                attempt += 1
                if attempt >= self.api_params['attempt_num']:
                    return None
                time.sleep(wait_sec)

    def multi_threading_openai_api_call(self, prompts, max_workers=64):
        timer = Timer()
        print(f"using model_{self.api_params['engine']}")
        print('Processing queires')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self.openai_api_call, enumerate(prompts, start=1)), 
                    total=len(prompts)
                )
            )
        print("Average time after {0} samples: {1}".format(len(prompts), timer.get_time(restart=False) / len(prompts)))
        print('Processed queries')
        return results

def response_extractor(response):
    llm_output = response.choices[0].message.content.strip()
    return {'llm_output': llm_output}

if __name__ == "__main__":
    prompts = [
        'Generate a story about a robot.',
        'Generate a story about a rabbit.',
        'Generate a story about a cat.',
        'Generate a story about a dog.',
        'Generate a story about a bird.',
        'Generate a story about a fish.',
        'Generate a story about a mouse.',
        'Generate a story about a horse.',
        'Generate a story about a cow.',
        'Generate a story about a pig.',
        'Generate a story about US president.',
        'Generate a story about a famous scientist.',
        'Generate a story about a famous artist.',
        'Generate a story about a famous writer.'
    ]
    openai_manager = OpenAIRequestManager(response_extractor)
    openai_manager.multi_threading_openai_api_call(prompts=prompts, max_workers=10)
   



