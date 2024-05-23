# **틴플레이 활동 추천 서비스**

## **💡목차**

1. 개요
2. 데이터 준비
3. CountVectorizer 사용
4. 코사인 유사도 계산
5. 결과
6. 느낀점

## **📋 개요**

활동 페이지 목록에서 게시글을 클릭하면 상세 페이지로 이동합니다.  
상세 페이지 하단에는 현재 보고 있는 활동(게시글)을 제외한 4개의 다른 활동이 표시됩니다.  
`활동 추천 서비스`는 해당 활동 게시글의 제목, 내용, 소개, 장소, 카테고리 데이터를 텍스트로 결합하고,  
이를 벡터화하여 `CountVectorizer`를 사용해 코사인 유사도를 계산하여 유사한 활동을 추천합니다.

<<전체 화면 이미지로 보여주기>>

<!-- <img src='./images/전체 화면.png'> -->

## **📊 데이터 준비 (Data Collection, Data Preprocessing)**

먼저, 활동 게시글의 데이터를 수집합니다.  
게시글 데이터는 제목, 내용, 소개, 장소, 카테고리 필드로 구성됩니다.
데이터는 `Pandas DataFrame`으로 관리합니다.

    @staticmethod
    def remove_html_tags(text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

> HTML 태그를 삭제하는 함수입니다.  
> 주어진 텍스트에서 HTML 태그를 정규표현식을 사용하여 찾아내고,  
> 해당 태그를 빈 문자열로 대체하여 삭제합니다.

    @staticmethod
    def remove_special_characters_except_spaces(text):
        """
        주어진 텍스트에서 숫자, 한글, 영어 알파벳을 제외한 모든 특수문자 및 기호를 제거하고,
        공백은 유지합니다.

        :param text: 특수문자 및 기호를 포함한 문자열
        :return: 특수문자 및 기호가 제거된 문자열 (공백 유지)
        """

        clean = re.compile('[^0-9a-zA-Zㄱ-ㅎ가-힣ㅏ-ㅣ ]')
        return re.sub(clean, ' ', text)

> 특수문자 및 기호를 제거하는 함수입니다.  
> 주어진 텍스트에서 숫자, 한글, 영어 알파벳을 제외한 모든 특수문자 및 기호를 찾아내고,  
> 이를 빈 문자열로 대체하여 제거합니다. 이때 공백은 유지됩니다.

<details>
  <summary>활동 데이터프레임 생성 코드</summary>

        # 활동 테이블에서 필요한 필드를 가져옵니다.
        activities = Activity.enabled_objects.annotate(
                category_name=F('category__category_name')
        ).values(
                'activity_title',
                'activity_content',
                'activity_intro',
                'activity_address_location',
                'id',
                'category_name'
        )

        # activity_data 리스트에 필요한 필드 값을 추가합니다.
        activity_data = []
        for activity in activities:
                activity_data.append(
                (
                        activity['activity_title'],
                        activity['activity_content'],
                        activity['activity_intro'],
                        activity['activity_address_location'],
                        activity['category_name'],
                        activity['id']
                )
                )

        # a_df 에 데이터 프레임을 생성합니다.
        a_df = pd.DataFrame(activity_data, columns=['activity_title', 'activity_content', 'activity_intro', 'activity_address_location', 'category_name', 'id'])

        a_df.activity_content = a_df.activity_content.apply(remove_html_tags)

        a_df.activity_content = a_df.activity_content.apply(lambda x: x.replace("\"", ""))

        a_df['feature'] = a_df['activity_title'] + ' ' + a_df['activity_content'] + ' ' + a_df['activity_intro'] + ' ' + a_df['activity_address_location'] + ' ' + a_df['category_name']

        a_df.feature = a_df.feature.apply(remove_special_characters_except_spaces)

        result_df = a_df.feature

</details>

## **📈 CountVectorizer 사용 (Text Vectorization)**

활동 게시글의 텍스트 데이터를 `CountVectorizer`를 사용하여 벡터화합니다.
텍스트 데이터는 제목, 내용, 소개, 장소, 카테고리 필드이며,  
위의 데이터 필드를 하나의 필드로 결합하여 벡터화합니다.

        from sklearn.feature_extraction.text import CountVectorizer

        count_v = CountVectorizer()
        count_metrix = count_v.fit_transform(ActivityDetailWebView.result_df)

<!-- <details>
  <summary>CountVectorizer</summary>
        from sklearn.feature_extraction.text import CountVectorizer

        count_v = CountVectorizer()
        count_metrix = count_v.fit_transform(ActivityDetailWebView.result_df)

</details> -->

## **📉 코사인 유사도 계산(Cosine Similarity Calculation)**

벡터화된 텍스트 데이터를 사용하여 `코사인 유사도`를 계산하고,  
현재 게시글을 제외한 유사도가 높은 상위 4개의 게시글을 가져옵니다.

        from sklearn.metrics.pairwise import cosine_similarity

        c_s = cosine_similarity(count_metrix)

<!-- <details>
  <summary>코사인 유사도 계산하기</summary>

        from sklearn.metrics.pairwise import cosine_similarity

        c_s = cosine_similarity(count_metrix)

</details> -->

## **📊 추천(Recommendation) 결과**

코사인 유사도 행렬을 기반으로,  
현재 보고 있는 게시글과 유사도가 높은 상위 4개의 게시글을 나열합니다.  
나열 순서는 왼쪽에서 오른쪽으로 보여집니다.

    @staticmethod
    def get_index_from_title(title):
        return ActivityDetailWebView.a_df[ActivityDetailWebView.a_df.feature == title].index[0]

> 활동 상세 페이지의 텍스트 데이터(title)를 통해 데이터 프레임(a_df)에서 동일한 feature의 인덱스를 찾아주는 함수입니다.  
> 해당 페이지의 텍스트 데이터(title)를 인자로 받아서 해당 데이터를 가진 행의 인덱스(index)를 반환합니다.  
> 즉, 주어진 텍스트 데이터에 해당하는 행의 인덱스를 찾는 역할을 합니다.

    @staticmethod
    def get_title_from_index(index):
        return ActivityDetailWebView.a_df[ActivityDetailWebView.a_df.index == index]['activity_title'].values[0]

> 특정 인덱스(index)에 해당하는 활동 제목(activity_title)을 찾는 함수입니다.
> 이 메서드는 인덱스(index)를 인자로 받아서 해당 인덱스를 가진 행의 활동 제목(activity_title)을 반환합니다.  
> 즉, 주어진 인덱스에 해당하는 행의 활동 제목을 찾는 역할을 합니다.

<details>
  <summary>활동 게시글 추천 결과를 반환하는 코드</summary>
  
        # 특정 활동의 상세 정보를 가져옵니다.
        detail_title = activity.activity_title
        detail_content = activity.activity_content
        detail_intro = activity.activity_intro
        detail_category = category.category_name
        detail_address = activity.activity_address_location

        # HTML 태그를 제거하고 텍스트 데이터를 결합합니다.
        remove_result = (
        self.remove_html_tags(detail_title) + ' ' +
        self.remove_html_tags(detail_content) + ' ' +
        self.remove_html_tags(detail_intro) + ' ' +
        self.remove_html_tags(detail_address) + ' ' +
        self.remove_html_tags(detail_category)
        )

        # 특수 문자를 제거합니다.
        similar_title = self.remove_special_characters_except_spaces(remove_result)

        # 제목을 기반으로 인덱스를 가져옵니다.
        similar_index = self.get_index_from_title(similar_title)

        # 코사인 유사도를 계산하여 유사한 활동을 정렬합니다.
        similar_activity_result = sorted(list(enumerate(cosine_sim[similar_index])), key=lambda x: x[1], reverse=True)

        all_activities = []  # 모든 활동을 저장할 리스트

        # 유사한 상위 4개의 활동을 리스트에 추가합니다.
        for similar_activity in similar_activity_result[1:5]:
        similar_activity_list = self.get_title_from_index(similar_activity[0])
        activity_items = similar_activity_list.splitlines()
        all_activities.extend(activity_items)

        # 추천 활동 목록에 표시할 활동들을 가져옵니다. 이때 현재 보고 있는 활동은 제외합니다.
        recommended_activities = list(
        Activity.enabled_objects.filter(activity_title__in=all_activities).exclude(id=activity_id)[:4]
        )

        # 관련이 높은 순서대로 다시 정렬합니다.
        recommended_activities = sorted(recommended_activities, key=lambda x: all_activities.index(x.activity_title))

</details>

<!-- <img src='./images/화면 하단.png'> -->

## **📌느낀점**

활동 추천 서비스의 목적

1. 사용자가 더 많은 활동에 관심을 가지게 하여 다양한 활동을 할 수 있도록 유도하는 것입니다. (사용자 유지율이 향상됩니다)
2. 다양한 활동을 하게 된다면 그 활동을 하는 모임에 참여할 가능성이 높아지고, 사용자의 활발한 참여도는 커뮤니티의 기대효과를 불러옵니다. (청년활동)
3. 활동 개설은 커뮤니티의 매출과 연관되어 있습니다. 활동 참여가 많아지고 더 많은 활동이 생긴다면 곧 커뮤니티의 매출 증대로 이어집니다. (매출 증대)

상세 페이지의 텍스트 데이터를 유사도 분석하여 추천할 때의 이점

1. 사용자가 현재 보고있는 데이터를 기반으로 한 분석 결과는 사용자가 선호할만한 항목을 더 정확하게 예측할 수 있습니다. (사용자의 깊은 인사이트를 얻을 수 있음)
2. 덜 인기있는 콘텐츠도 유사도 분석을 통해 더 많은 노출 기회를 얻게 됩니다. 이는 콘텐츠 플랫폼의 전체적인 활성화에 기여할 수 있습니다. (롱테일 효과)
3. 특정 텍스트에 관심이 있는 사용자를 타겟으로 삼은 효율적인 마케팅이 될 수 있습니다. (탐색 피로도 감소)
