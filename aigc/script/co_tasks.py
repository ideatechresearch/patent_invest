QUERY_TASKS = [
    {
        "url_template": "https://{host}/cloudidp/api/company-exception-list?key={api_key}&keyWord={keyword}",
        "desc": "经营异常名录",
        "question": "企业是否存在经营异常记录？若有，列出异常原因与时间。",
        "field_mapping": {
            "total": "总记录条数",

            # 嵌套字段 data.items
            "items.name": "单位名称",
            "items.unityCreditCode": "统一社会信用代码",
            "items.type": "异常类型",
            "items.inReason": "列入原因",
            "items.inDate": "列入日期",
            "items.inOrgan": "列入决定机构",
            "items.outReason": "移出原因",
            "items.outDate": "移出日期",
            "items.outOrgan": "移出决定机构"
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/saic-basic-info?key={api_key}&keyWord={keyword}",
        "desc": "工商基本信息",
        "question": "请判断企业基本注册信息中是否存在异常，例如注册资本异常波动、频繁变更法人或地址等情况。",
        "field_mapping": {
            # 顶层字段
            "name": "单位名称",
            "historyName": "历史名称",
            "registNo": "注册号",
            "unityCreditCode": "统一社会信用代码",
            "type": "单位类型",
            "legalPerson": "法定代表人",
            "registFund": "注册资本",
            "openDate": "成立日期",
            "startDate": "营业期限起始日期",
            "endDate": "营业期限终止日期",
            "registOrgan": "登记机关",
            "licenseDate": "核准日期",
            "state": "登记状态",
            "address": "住所地址",
            "scope": "经营范围",
            "revokeDate": "注销或吊销日期",
            "isOnStock": "是否上市",
            "priIndustry": "国民经济行业分类大类名称",
            "subIndustry": "国民经济行业分类小类名称",
            "legalPersonSurname": "法人姓",
            "legalPersonName": "法人名",
            "registCapital": "注册资金",
            "registCurrency": "注册资金币种",
            "industryCategoryCode": "行业门类代码",
            "industryLargeClassCode": "行业大类代码",
            "typeCode": "企业类型代码",
            "country": "国家",
            "province": "省",
            "city": "市",
            "area": "区/县",
            "lastUpdateDate": "最近更新日期",

            # 嵌套字段 data.partners
            "partners.name": "股东名称",
            "partners.type": "股东类型",
            "partners.identifyType": "证照/证件类型",
            "partners.identifyNo": "证照/证件号码",
            "partners.shouldType": "认缴出资方式",
            "partners.shouldCapi": "认缴出资金额",
            "partners.shoudDate": "认缴出资日期",
            "partners.realType": "实缴出资方式",
            "partners.realCapi": "实缴出资额",
            "partners.realDate": "实缴出资日期",

            # 嵌套字段 data.employees
            "employees.name": "姓名",
            "employees.job": "职务",

            # 嵌套字段 data.branchs
            "branchs.name": "分支机构名称",
            "branchs.brRegNo": "分支机构企业注册号",
            "branchs.brPrincipal": "分支机构负责人",
            "branchs.cbuItem": "一般经营项目",
            "branchs.brAddr": "分支机构地址",

            # 嵌套字段 data.changes
            "changes.changesType": "变更事项",
            "changes.changesBeforeContent": "变更前内容",
            "changes.changesAfterContent": "变更后内容",
            "changes.changesDate": "变更日期"
        },
        "field_path": ["data"]
    },
    {
        "url_template": 'https://{host}/cloudidp/outer/exactSaicInfo?key={api_key}&keyWord={keyword}',
        "desc": "工商全维度信息",
        "question": '''请基于企业在工商系统中的全量信息（包括注册信息、股东结构、历史变更、年报数据、行政处罚、经营异常、对外投资与担保等），从以下角度综合判断企业是否存在潜在经营或合规风险：
        1. 基本信息是否稳定，是否存在频繁的注册资本变动、法定代表人或注册地址变更；
        2. 股东结构是否复杂，是否频繁变更股东或实控人；
        3. 是否存在行政处罚记录、列入经营异常或严重违法失信名单等情形；
        4. 年报数据是否合理，包括：从业人数为 0、营业收入为负、长期亏损、社保与纳税金额为 0 或与规模不符；
        5. 是否存在对外投资/担保过多，形成复杂的关联链条或疑似壳公司结构；
        6. 是否存在分支机构异常、短期设立多个分支、经营范围与主业偏离等特征；
        请结合企业年报信息与工商登记数据是否一致，分析其经营合规性与潜在风险。"
        ''',
        'field_mapping': {
            "name": "单位名称",
            "historyName": "历史名称",
            "registNo": "注册号",
            "unityCreditCode": "统一社会信用代码",
            "saicCode": "全国企业信用信息公示系统代码",
            "type": "单位类型",
            "legalPerson": "法定代表人",
            "legalPersonType": "法人类型",
            "registFund": "注册资本",
            "registFundCurrency": "注册资金币种",
            "openDate": "成立日期",
            "businessMan": "经营者",
            "startDate": "营业期限起始日期",
            "endDate": "营业期限终止日期",
            "registOrgan": "登记机关",
            "licenseDate": "核准日期",
            "state": "登记状态",
            "address": "住所地址",
            "scope": "经营范围",
            "revokeDate": "注销或吊销日期",
            "notice": "简易注销公告",
            "isOnStock": "是否上市",
            "priIndustry": "行业门类描述",
            "industryCategoryCode": "行业门类代码",
            "subIndustry": "行业大类描述",
            "industryLargeClassCode": "行业大类代码",
            "middleCategory": "行业中类描述",
            "middleCategoryCode": "行业中类code",
            "smallCategory": "行业小类描述",
            "smallCategoryCode": "行业小类code",
            "province": "省",
            "ancheYear": "最后年检年度",
            "ancheYearDate": "最后年检年度日期",
            "contactPhone": "联系电话",
            "contactEmail": "联系邮箱",
            "lastUpdateDate": "最近更新日期",

            # 嵌套字段 - 股东信息
            "stockholders.name": "股东名称",
            "stockholders.type": "股东类型",
            "stockholders.strType": "股东类型",
            "stockholders.identifyType": "证照/证件类型",
            "stockholders.identifyNo": "证照/证件号码",
            "stockholders.investType": "认缴出资方式",
            "stockholders.subconam": "认缴出资金额",
            "stockholders.conDate": "认缴出资日期",
            "stockholders.realType": "实缴出资方式",
            "stockholders.realAmount": "实缴出资额",
            "stockholders.realDate": "实缴出资日期",
            "stockholders.regCapCur": "币种",
            "stockholders.fundedRatio": "出资比例",

            # 嵌套字段 - 主要成员
            "employees.name": "姓名",
            "employees.job": "职务",
            "employees.sex": "性别",
            "employees.type": "职位类别",

            # 嵌套字段 - 分支机构
            "branchs.name": "分支机构名称",
            "branchs.brRegNo": "分支机构企业注册号",
            "branchs.brPrincipal": "分支机构负责人",
            "branchs.cbuItem": "一般经营项目",
            "branchs.brAddr": "分支机构地址",

            # 嵌套字段 - 变更记录
            "changes.type": "变更内容",
            "changes.beforeContent": "变更前内容",
            "changes.afterContent": "变更后内容",
            "changes.changeDate": "变更日期",

            # 嵌套字段 - 经营异常
            "changemess.inreason": "列入经营异常名录原因",
            "changemess.indate": "列入日期",
            "changemess.outreason": "移出经营异常名录原因",
            "changemess.outdate": "移出日期",
            "changemess.belongorg": "作出列入决定机关",
            "changemess.outOrgan": "作出移出决定机关",

            # 嵌套字段 - 年报
            "reports.annualreport": "报送年度",
            "reports.releasedate": "发布日期",

            # 嵌套字段 - 失信黑名单
            "illegals.order": "序号",
            "illegals.type": "类别",
            "illegals.reason": "列入原因",
            "illegals.date": "列入日期",
            "illegals.organ": "做出决定机关（列入）",
            "illegals.reasonOut": "移出原因",
            "illegals.dateOut": "移出日期",
            "illegals.organOut": "作出决定机关（移出）",
        },
        "field_path": ["result"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/company-out-investment?key={api_key}&keyWord={keyword}",
        "desc": "对外投资情况",
        "question": "企业是否存在对外投资？若有，请分析投资行业是否集中，是否涉及高风险领域。",
        'field_mapping': {
            "name": "公司名称",
            "no": "注册号",
            "creditCode": "全国统一信用代码",
            "econKind": "企业类型",
            "status": "状态",
            "registCapi": "注册资本",
            "operName": "法人",
            "fundedRatio": "出资比例",
            "startDate": "成立日期"
        },
        "field_path": ["data", "companyOutInvestment"],

    },
    {
        "url_template": "https://{host}/cloudidp/api/base-account-record?key={api_key}&keyWord={keyword}",
        "desc": "基本账户履历",
        "question": "请分析企业基本账户是否存在频繁变更、账户被撤销等风险迹象。",
        'field_mapping': {
            "total": "总记录条数",

            # items 是列表中的字段，采用 items.xxx 的格式表示
            "items.name": "单位名称/账户名称",
            "items.licenseKey": "基本户许可证号",
            "items.licenseOrg": "审批机关",
            "items.licenseDate": "审批日期",
            "items.licenseType": "许可类型"
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/annual-report-info?key={api_key}&keyWord={keyword}",
        "desc": "企业年报信息",
        "question": '''请结合企业性质（如是否为上市公司、所属行业）核查其年报信息，关注以下方面，并简要说明潜在问题及其可能原因或风险：

        一、年报披露合规性  
        1. 是否存在年报缺失或未按时披露的情况（如连续年份缺报、严重滞后等）？  
        2. 对于上市公司、重点行业或大型企业，如存在连续多年未披露核心信息（如财务数据），是否存在信息披露机制缺失或合规风险？  
        
        二、财务信息异常  
        3. 是否存在明显异常的财务数据（如营业收入为负、连续多年亏损、资产负债比例失衡等）？  
        4. 财务数据趋势是否与企业规模、行业特性相符？是否存在突变或长期异常？
        
        三、工商信息一致性  
        5. 年报中披露的从业人数、注册资本、对外投资、社保缴纳人数等关键指标，是否与工商登记信息一致？是否存在虚报/漏报现象？
        
        
        ⚠️ 禁止性判断说明（适用于小微企业、未强制披露企业）  
        
        在判断年报问题时，需注意以下禁止性结论，避免误判：
        
        1. **不可直接将缺失视为风险**：  
           年报中未披露以下字段（如从业人数、总资产、总负债、营业收入、净利润、主营业务收入、总利润、税收金额、所有者权益）时，可能是因行业申报差异、政策豁免、数据未强制披露等，不应直接视为异常。
        2. **不得以“参保人数为 0”直接推断无员工/经营异常**：  
           参保人数为 0 有可能因未及时填报、特殊行业（如纯外包、外籍员工）或个体工商性质，不应作为独立风险依据。
        3. **需结合上下文与行业惯例**：  
           对于缺失或异常情况，应综合企业规模、业务体量、所属行业、披露年限等维度判断。  
           例如：头部房企、上市公司若连续多年不披露资产负债表，可能构成信息披露异常信号。
        
        ### ✅ 使用建议：
        - 对**上市公司/重点行业企业**：请严格对照上述所有条款进行核查；信息披露完整性是重点。  
        - 对**小微企业/个体户等**：可适度放宽要求，如发现企业连续多年（如 5 年及以上）未披露任何核心财务信息（如资产、负债、利润），可作为信息披露异常的信号，重点判断是否符合行业惯例或合理推断标准。
        ''',
        'field_mapping': {
            "year": "报送年度",
            "remarks": "备注",
            "hasDetailInfo": "是否有详细信息",
            "publishDate": "发布日期",

            # basicInfoData 列表里的字段
            "basicInfoData.regNo": "注册号",
            "basicInfoData.companyName": "企业名称",
            "basicInfoData.operatorName": "经营者姓名",
            "basicInfoData.contactNo": "企业联系电话",
            "basicInfoData.postCode": "邮政编码",
            "basicInfoData.address": "企业通信地址",
            "basicInfoData.emailAddress": "电子邮箱",
            "basicInfoData.isStockRightTransfer": "有限责任公司本年度是否发生股东股权转让",
            "basicInfoData.status": "企业经营状态",
            "basicInfoData.hasWebSite": "是否有网站或网店",
            "basicInfoData.hasNewStockOrByStock": "企业是否有投资信息或购买其他公司股权",
            "basicInfoData.employeeCount": "从业人数",
            "basicInfoData.belongTo": "隶属关系",
            "basicInfoData.capitalAmount": "资金数额",
            "basicInfoData.hasProvideAssurance": "是否有对外担保信息",
            "basicInfoData.operationPlaces": "经营场所",
            "basicInfoData.mainType": "主体类型",
            "basicInfoData.operationDuration": "经营期限",
            "basicInfoData.ifContentSame": "章程信息(是否一致)",
            "basicInfoData.differentContent": "章程信息(不一致内容)",
            "basicInfoData.generalOperationItem": "经营范围(一般经营项目)",
            "basicInfoData.approvedOperationItem": "经营范围(许可经营项目)",

            # assetsData
            "assetsData.totalAssets": "资产总额",
            "assetsData.totalOwnersEquity": "所有者权益合计",
            "assetsData.grossTradingIncome": "营业总收入",
            "assetsData.totalProfit": "利润总额",
            "assetsData.mainBusinessIncome": "营业总收入中主营业务",
            "assetsData.netProfit": "净利润",
            "assetsData.totalTaxAmount": "纳税总额",
            "assetsData.totalLiabilities": "负债总额",
            "assetsData.bankingCredit": "金融贷款",
            "assetsData.governmentSubsidy": "获得政府扶持资金、补助",

            # changeList
            "changeList.no": "序号",
            "changeList.changeName": "修改事项",
            "changeList.before": "变更前内容",
            "changeList.after": "变更后内容",
            "changeList.changeDate": "变更日期",

            # investInfoList
            "investInfoList.no": "序号",
            "investInfoList.name": "投资设立企业或购买股权企业名称",
            "investInfoList.regNo": "注册号",
            "investInfoList.shouldCapi": "认缴出资额（万元）",
            "investInfoList.shareholdingRatio": "持股比例（%）",

            # partnerList
            "partnerList.no": "序号",
            "partnerList.name": "股东/发起人",
            "partnerList.shouldCapi": "认缴出资额",
            "partnerList.shouldDate": "认缴出资时间",
            "partnerList.shouldType": "认缴出资方式",
            "partnerList.realCapi": "实缴出资额",
            "partnerList.realDate": "实缴出资时间",
            "partnerList.realType": "实缴出资方式",
            "partnerList.form": "出资类型",
            "partnerList.investmentRatio": "出资比例",

            # provideAssuranceList
            "provideAssuranceList.no": "序号",
            "provideAssuranceList.creditor": "债权人",
            "provideAssuranceList.debtor": "债务人",
            "provideAssuranceList.creditorCategory": "主债权种类",
            "provideAssuranceList.creditorAmount": "主债权数额",
            "provideAssuranceList.fulfillObligation": "履行债务的期限",
            "provideAssuranceList.assuranceDurn": "保证的期间",
            "provideAssuranceList.assuranceType": "保证的方式",
            "provideAssuranceList.assuranceScope": "保证担保的范围",

            # stockChangeList
            "stockChangeList.no": "序号",
            "stockChangeList.name": "股东",
            "stockChangeList.before": "变更前股权比例",
            "stockChangeList.after": "变更后股权比例",
            "stockChangeList.changeDate": "股权变更日期",

            # webSiteList
            "webSiteList.no": "序号",
            "webSiteList.type": "类型",
            "webSiteList.name": "名称",
            "webSiteList.webSite": "网址",

            # administrationLicenseList
            "administrationLicenseList.no": "序号",
            "administrationLicenseList.name": "许可文件名称",
            "administrationLicenseList.endDate": "有效期至",

            # branchList
            "branchList.name": "分支机构名称",
            "branchList.regNo": "注册号",
            "branchList.address": "住所",
            "branchList.principal": "负责人",

            # employeeList
            "employeeList.no": "序号",
            "employeeList.name": "名称",
            "employeeList.job": "职位",
            "employeeList.cerNo": "证照/证件号码",
            "employeeList.scertName": "证件名称",
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/company-black-list?key={api_key}&keyWord={keyword}",
        "desc": "严重违法信息",
        "question": "企业是否被列入工商严重违法失信名单？若有，请说明原因及时间。",
        'field_mapping': {
            "total": "总记录条数",
            "items.name": "单位名称",
            "items.unityCreditCode": "统一社会信用代码",
            "items.type": "类型",
            "items.inReason": "列入原因",
            "items.inDate": "列入日期",
            "items.inOrgan": "列入决定机构",
            "items.outReason": "移出原因",
            "items.outDate": "移出日期",
            "items.outOrgan": "移出决定机构",
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/outer/finalBeneficiary?key={api_key}&keyWord={keyword}",
        "desc": "最终受益人识别",
        "question": "请说明企业的最终受益人结构是否复杂，是否涉及疑似隐名股东或关联人。",
        'field_mapping': {
            # 顶层字段
            "updateTime": "数据更新时间",
            "company": "目标企业",
            "finalBeneficiary": "最终受益人列表",

            # 嵌套字段（result.finalBeneficiary 为列表）
            "finalBeneficiary.name": "受益人名称",
            "finalBeneficiary.type": "受益人类型",
            "finalBeneficiary.identifyType": "证照/证件类型",
            "finalBeneficiary.identifyNo": "证照/证件号码",
            "finalBeneficiary.capital": "出资额",
            "finalBeneficiary.capitalPercent": "出资比例",
            "finalBeneficiary.capitalChain": "出资链"
        },
        "field_path": ["result"]
    },
    {
        "url_template": "https://{host}/cloudidp/outer/equityShareList?key={api_key}&keyWord={keyword}",
        "desc": "股权结构",
        "question": "请分析企业股权结构是否集中，是否存在交叉持股或频繁变更现象。",
        'field_mapping': {
            # 顶层字段
            "total": "总记录条数",

            # 嵌套列表字段：result.item
            "item.name": "企业名称或者人名",
            "item.type": "股东类型",
            "item.layer": "层级",
            "item.percent": "股份占比",
            "item.capital": "出资额",
            "item.parent": "父亲节点"
        },
        "field_path": ["result"]
    },
    # {
    #     "url_template": "https://{host}/cloudidp/api/identity/microEnt?key={api_key}&keyWord={keyword}",
    #     "desc": "小微企业识别",
    #     "question": "该企业是否为小微企业？如是，请简要说明识别依据。"
    # },
    {
        "url_template": "https://{host}/cloudidp/api/simple-cancellation?key={api_key}&keyWord={keyword}",
        "desc": "简易注销公告",
        "question": "企业是否已申请简易注销？若有，请说明公告信息。",
        "field_mapping": {
            # 顶层字段
            "name": "企业名称",
            "unityCreditCode": "统一社会信用代码",
            "registNo": "注册号",
            "registOrgan": "登记机关",
            "publicDate": "公告期",
            "docUrl": "全体投资人承诺书Url",

            # 嵌套列表字段 data.dissentInfos
            "dissentInfos.dissentPerson": "异议申请人",
            "dissentInfos.dissentContent": "异议内容",
            "dissentInfos.dissentDate": "异议时间",

            # 嵌套列表字段 data.cancellationResults
            "cancellationResults.cancellationContent": "简易注销内容",
            "cancellationResults.publicDate": "公告申请日期"
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/court-notice?key={api_key}&keyWord={keyword}",
        "desc": "开庭公告",
        "question": "是否存在开庭公告？若有，请列出案件简要信息、开庭时间，案由",
        "field_mapping": {
            "pageSize": "第几页",
            "pageIndex": "每页条数",
            "totalRecords": "总数",
            "result.defendant": "被告/被上诉人",
            "result.executegov": "法院",
            "result.prosecutor": "原告/上诉人",
            "result.courtDate": "开庭日期",
            "result.caseReason": "案由",
            "result.caseNo": "案号"
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/judgment-doc?key={api_key}&keyWord={keyword}",
        "desc": "裁判文书",
        "question": "企业是否涉及已判决的法律纠纷？若有，请简述判决内容与案件类型等关键信息。",
        'field_mapping': {
            "court": "执行法院",
            "caseNo": "裁判文书编号",
            "caseType": "裁判文书类型",
            "submitDate": "提交时间",
            "updateDate": "修改时间",
            "isProsecutor": "是否原告",
            "isDefendant": "是否被告",
            "courtYear": "开庭时间年份",
            "caseRole": "涉案人员角色",
            "companyName": "公司名称",
            "title": "裁判文书标题",
            "sortTime": "审结时间",
            "body": "内容",
            "caseCause": "案由",
            "judgeResult": "裁决结果",
            "yiju": "依据",
            "judge": "审判员",
            "trialProcedure": "审理程序",
        },
        "field_path": ["data", "data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/court-announcement?key={api_key}&keyWord={keyword}",
        "desc": "法院公告",
        "question": "是否存在法院公告信息？若有，请简要说明公告内容。",
        'field_mapping': {
            "court": "执行法院",
            "companyName": "公司名称",
            "sortTime": "发布时间",
            "body": "内容",
            "relatedParty": "相关当事人",
            "ggType": "公告类型",  # 裁判文书,起诉状副本及开庭传票
        },
        "field_path": ["data", "data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/dishonesty?key={api_key}&keyWord={keyword}",
        "desc": "失信信息",
        "question": "企业是否被列入失信被执行人名单？如有，请说明失信行为。",
        'field_mapping': {
            "province": "所在省份缩写",
            "yiwu": "生效法律文书确定的义务",
            "updatedate": "记录更新时间",
            "performedpart": "已履行",
            "unperformpart": "未履行",
            "orgType": "组织类型（1：自然人，2：企业，3：社会组织，空白：无法判定）",
            "orgTypeName": "组织类型名称",
            "companyName": "公司名称",
            "pname": "被执行人姓名",
            "sortTime": "立案时间",
            "court": "执行法院名称",
            "lxqk": "被执行人的履行情况",
            "yjCode": "执行依据文号",
            "idcardNo": "身份证/组织机构代码证",
            "yjdw": "做出执行依据单位",
            "jtqx": "失信被执行人行为具体情形",
            "caseNo": "案号",
            "postTime": "发布时间",
            "ownerName": "法定代表人或者负责人姓名",
        },
        "field_path": ["data", "data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/implements?key={api_key}&keyWord={keyword}",
        "desc": "被执行信息",
        "question": "企业是否存在被执行记录？若有，请说明执行金额和执行法院。",
        'field_mapping': {
            "companyName": "公司名称",
            "pname": "被执行姓名",
            "sortTime": "立案时间",
            "caseNo": "案号",
            "court": "执行法院名称",
            "execMoney": "执行标的",
            "idcardNo": "身份证/组织机构代码",
        },
        "field_path": ["data", "data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/stock-freeze?key={api_key}&keyWord={keyword}",
        "desc": "股权冻结信息",
        "question": "企业是否存在股权被冻结情况？请说明冻结股东及金额。",
        'field_mapping': {
            "executedBy": "被执行人",
            "equityAmount": "股权数额",
            "enforcementCourt": "执行法院",
            "executionNoticeNum": "执行通知书文号",
            "status": "状态",  # 股权冻结|冻结

            "equityFreezeDetail": "股权冻结情况",
            "equityUnFreezeDetail": "解除冻结详情",
            "judicialPartnersChangeDetail": "股东变更信息",
            # 二级字段 - 股权冻结情况
            "equityFreezeDetail.companyName": "冻结-相关企业名称",
            "equityFreezeDetail.executionMatters": "冻结-执行事项",
            "equityFreezeDetail.executionDocNum": "冻结-执行文书文号",
            "equityFreezeDetail.executionVerdictNum": "冻结-执行裁定书文号",
            "equityFreezeDetail.executedPersonDocType": "冻结-被执行人证件种类",
            "equityFreezeDetail.executedPersonDocNum": "冻结-被执行人证件号码",
            "equityFreezeDetail.freezeStartDate": "冻结-开始日期",
            "equityFreezeDetail.freezeEndDate": "冻结-结束日期",
            "equityFreezeDetail.freezeTerm": "冻结-冻结期限",
            "equityFreezeDetail.publicDate": "冻结-公示日期",

            # 二级字段 - 解除冻结详情
            "equityUnFreezeDetail.executionMatters": "解除-执行事项",
            "equityUnFreezeDetail.executionVerdictNum": "解除-执行裁定书文号",
            "equityUnFreezeDetail.executionDocNum": "解除-执行文书文号",
            "equityUnFreezeDetail.executedPersonDocType": "解除-被执行人证件种类",
            "equityUnFreezeDetail.executedPersonDocNum": "解除-被执行人证件号码",
            "equityUnFreezeDetail.unFreezeDate": "解除-解除冻结日期",
            "equityUnFreezeDetail.publicDate": "解除-公示日期",
            "equityUnFreezeDetail.thawOrgan": "解除-解冻机关",
            "equityUnFreezeDetail.thawDocNo": "解除-解冻文书号",

            # 二级字段 - 股东变更信息
            "judicialPartnersChangeDetail.executionMatters": "变更-执行事项",
            "judicialPartnersChangeDetail.executionVerdictNum": "变更-执行裁定书文号",
            "judicialPartnersChangeDetail.executedPersonDocType": "变更-被执行人证件种类",
            "judicialPartnersChangeDetail.executedPersonDocNum": "变更-被执行人证件号码",
            "judicialPartnersChangeDetail.assignee": "变更-受让人",
            "judicialPartnersChangeDetail.assistExecDate": "变更-协助执行日期",
            "judicialPartnersChangeDetail.assigneeDocKind": "变更-受让人证件种类",
            "judicialPartnersChangeDetail.assigneeRegNo": "变更-受让人证件号码",
            "judicialPartnersChangeDetail.stockCompanyName": "变更-股权所在公司名称"
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/case-filing?key={api_key}&keyWord={keyword}",
        "desc": "立案信息",
        "question": "是否存在立案记录？若有，请说明案件类型和立案时间。",
        'field_mapping': {
            "pageSize": "每页条数",
            "pageIndex": "第几页",
            "totalRecords": "总记录数",
            "result": "案件详情",
            "result.caseNo": "案号",
            "result.publishDate": "立案日期",
            "result.courtYear": "案件年份",
            "result.prosecutorList": "公诉人/原告/上诉人/申请人列表",
            "result.prosecutorList.name": "原告名称",
            "result.defendantList": "被告人/被告/被上诉人/被申请人列表",
            "result.defendantList.name": "被告名称",
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/shellCompany?key={api_key}&keyWord={keyword}",
        "desc": "空壳企业识别",
        "question": "该企业是否疑似为空壳企业？请说明判断依据。",
        'field_mapping': {
            "hasShellTags": "是否空壳",
            "shellTagsCount": "空壳特征数量",
            "tagsDetail": "空壳特征明细",
            "tagsDetail.tagAlias": "特征简称",
            "tagsDetail.tagTitle": "特征标题",
            "tagsDetail.description": "情况描述",
        },
        "field_path": ["data"]
    },
    {
        "url_template": "https://{host}/cloudidp/api/company-stock-relation?key={api_key}&keyWord={keyword}&name={person}",
        "desc": "人员对外投资信息",
        "question": "请对该企业法人担任法人的前五家企业进行深度分析，包括这些企业的基本信息、经营状况等，判断法人是否持有多家空壳公司，评估是否存在关联交易风险、利益输送风险或其他潜在风险。",
        'field_mapping': {
            "companyLegal": "担任法人公司信息",
            "foreignInvestments": "对外投资信息",
            "foreignOffices": "在外任职信息",

            # companyLegal 担任法人公司信息
            "companyLegal.name": "担任法人企业名称",
            "companyLegal.regNo": "担任法人企业注册号",
            "companyLegal.regCap": "担任法人企业注册资本",
            "companyLegal.regCapCur": "担任法人企业注册资本币种",
            "companyLegal.status": "担任法人企业状态",
            "companyLegal.ecoKind": "担任法人企业类型",

            # foreignInvestments 对外投资信息
            "foreignInvestments.name": "对外投资企业名称",
            "foreignInvestments.regNo": "对外投资企业注册号",
            "foreignInvestments.regCap": "对外投资企业注册资本",
            "foreignInvestments.regCapCur": "对外投资企业注册资本币种",
            "foreignInvestments.status": "对外投资企业状态",
            "foreignInvestments.ecoKind": "对外投资企业类型",
            "foreignInvestments.subConAmt": "认缴出资额",
            "foreignInvestments.subCurrency": "认缴出资币种",

            # foreignOffices 在外任职信息
            "foreignOffices.name": "在外任职企业名称",
            "foreignOffices.regNo": "在外任职企业注册号",
            "foreignOffices.regCap": "在外任职企业注册资本",
            "foreignOffices.regCapCur": "在外任职企业注册资本币种",
            "foreignOffices.status": "在外任职企业状态",
            "foreignOffices.ecoKind": "在外任职企业类型",
            "foreignOffices.position": "在外任职职位"
        },
        "field_path": ["data"],
        'exec': 'company_stock_deep_relation',
    },
    # {
    #     "url_template": "https://{host}/cloudidp/api/tax-arrears-info?key={api_key}&keyWord={keyword}",
    #     "desc": "欠税信息",
    #     "question": "企业是否存在欠税记录？如有，涉及哪些税种与欠税余额？",
    #     "field_mapping":
    #         {
    #             "overduePeriod": "欠税所属期",
    #             "pubDepartment": "发布单位",
    #             "taxpayerType": "纳税人类型",
    #             "pubDate": "发布日期",
    #             "area": "所属市县区",
    #             "address": "经营地点",
    #             "operName": "负责人姓名",
    #             "taxpayerNum": "纳税人识别号",
    #             "overdueAmount": "欠税余额",
    #             "overdueType": "欠税税种",
    #             "operIdNum": "企业证照号",
    #             "currOverdueAmount": "当前发生的欠税余额",
    #         },
    #     "field_path": ["data"]
    # },
    # {
    #     "url_template": "https://{host}/cloudidp/api/tax-case?key={api_key}&keyWord={keyword}",
    #     "desc": "税收违法",
    #     "question": "企业是否存在税收违法信息？如有，涉及案件性质是什么？",
    #     "field_mapping":
    #         {
    #             "property": "案件性质",  # 税收异常非正常户
    #             "name": "纳税人名称",
    #             "time": "发生时间"
    #         },
    #     "field_path": ["data"]
    # },
]
