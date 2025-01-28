import openai
import pandas as pd
import faker
from openai import OpenAI
import numpy as np
import re



client = OpenAI(api_key='your-api-key')






def generate_conversation():
    system_prompt = """
                        Generate a word conversation or report with a maximum of 250 words that contains various placeholders for personal identifiable information (PII). Use the following placeholders:
                        - CREDIT_CARD: <<CREDIT_CARD>>
                        - DATE_TIME: <<DATE_TIME>>
                        - EMAIL_ADDRESS: <<EMAIL_ADDRESS>>
                        - IBAN_CODE: <<IBAN_CODE>>
                        - IP_ADDRESS: <<IP_ADDRESS>>
                        - NRP: <<NRP>>
                        - LOCATION: <<LOCATION>>
                        - PERSON: <<PERSON>>
                        - PHONE_NUMBER: <<PHONE_NUMBER>>
                        - URL: <<URL>>
                        - US_BANK_NUMBER: <<US_BANK_NUMBER>>
                        - US_DRIVER_LICENSE: <<US_DRIVER_LICENSE>>
                        - US_ITIN: <<US_ITIN>>
                        - US_PASSPORT: <<US_PASSPORT>>
                        - US_SSN: <<US_SSN>>
                    """
    user_prompt = """Please talk to the customer servive representative about your concern -  regarding any topic that includes personal information.  Make it a maximum of 250 words. You must incorporate all/most/some of the fake personal information in the {Report}. Most/Some/All the below personal identifiable information must be incorporated with fake data:
                    - {CREDIT_CARD}
                    - {DATE_TIME}
                    - {EMAIL_ADDRESS}
                    - {IBAN_CODE}
                    - {IP_ADDRESS}
                    - {NRP}
                    - {LOCATION}
                    - {PERSON}
                    - {PHONE_NUMBER}
                    - {URL}
                    - {US_BANK_NUMBER}
                    - {US_DRIVER_LICENSE}
                    - {US_ITIN}
                    - {US_PASSPORT}
                    - {US_SSN}
                """
    prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        
    response = client.chat.completions.create(
                        model="gpt-4o", 
                        messages=prompt,
                        max_tokens=500,
                    )
   
    return response.choices[0].message.content


fake = faker.Faker()

faker_replacements = {
                        '<<CREDIT_CARD>>': fake.credit_card_number,
                        '<<DATE_TIME>>': lambda: str(fake.date_time()),
                        '<<EMAIL_ADDRESS>>': fake.email,
                        '<<IBAN_CODE>>': fake.iban,
                        '<<IP_ADDRESS>>': fake.ipv4,
                        '<<NRP>>': lambda: fake.random_element(elements=['American', 'Christian', 'Democrat']),  # Callable function
                        '<<LOCATION>>': fake.city,
                        '<<PERSON>>': fake.name,
                        '<<PHONE_NUMBER>>': fake.phone_number,
                        '<<URL>>': fake.url,
                        '<<US_BANK_NUMBER>>': fake.bban,
                        '<<US_DRIVER_LICENSE>>': fake.license_plate,
                        '<<US_ITIN>>': lambda: '9' + fake.random_element(['7', '8']) + str(fake.random_number(digits=7)),
                        '<<US_PASSPORT>>': lambda: str(fake.random_number(digits=9)),
                        '<<US_SSN>>': fake.ssn
                    }

def custom_tokenize(text):
    pattern = r'<<\w+>>|\w+|[^\w\s]'
    return re.findall(pattern, text)

def replace_placeholders_and_generate_bio(conversation_with_placeholders):
    tokens_with_fakes = []
    bio_tags = []
    
    placeholder_tokens = custom_tokenize(conversation_with_placeholders)

    for token in placeholder_tokens:
        if token in faker_replacements:
            
            fake_value = faker_replacements[token]()
            fake_tokens = fake_value.split(' ')
            
            tokens_with_fakes.extend(fake_tokens)
        
            bio_tags.append(f"B-{token[2:-2]}")  
            bio_tags.extend([f"I-{token[2:-2]}"] * (len(fake_tokens) - 1))  

        else:
            tokens_with_fakes.append(token)
            bio_tags.append("O")
    
    return ' '.join(tokens_with_fakes), bio_tags, tokens_with_fakes


def create_dataset(num_samples):
    data = []
    
    for _ in range(num_samples):
        conversation_with_placeholders = generate_conversation()
        
        conversation_with_fakes, bio_tags, tokens_with_fakes = replace_placeholders_and_generate_bio(conversation_with_placeholders)
        
        data.append([conversation_with_fakes, tokens_with_fakes, bio_tags])
  
    df = pd.DataFrame(data, columns=['Data', 'tokens','Label'])
    df['document'] = df.index + 1
    df.to_excel('pii_dataset_bio_test.xlsx', index=False)
    json_data = df.to_json(orient='records', indent=4)

    with open('./data/test_data.json', 'w') as f:
        f.write(json_data)

n = 1
create_dataset(n)
