from utils import *
from config import Config

Ideatech_Host = 'matrix.ideatech.info'
# Config.load('../config.yaml')

async def exact_saic_info(keyword, time_out=100):
    '''查询工商全维度信息（年检比对等批量调用场景使用）'''
    url = f'https://{Ideatech_Host}/cloudidp/outer/exactSaicInfo?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    field_mapping = {
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
    }
    return map_fields(data.get("result", {}), field_mapping)


async def real_time_saic_info(keyword, time_out=100):
    '''查询工商全维度信息（开户等单笔查询场景使用）'''
    url = f'https://{Ideatech_Host}/cloudidp/outer/realTimeSaicInfo?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    field_mapping = {
        # 一级字段
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

        # 股东信息列表
        "stockholders.name": "股东名称",
        "stockholders.type": "股东类型",
        "stockholders.strType": "股东类型",
        "stockholders.identifyType": "证照类型",
        "stockholders.identifyNo": "证照号码",
        "stockholders.investType": "认缴出资方式",
        "stockholders.subconam": "认缴出资金额",
        "stockholders.conDate": "认缴出资日期",
        "stockholders.realType": "实缴出资方式",
        "stockholders.realAmount": "实缴出资额",
        "stockholders.realDate": "实缴出资日期",
        "stockholders.regCapCur": "币种",
        "stockholders.fundedRatio": "出资比例",

        # 主要成员列表
        "employees.name": "姓名",
        "employees.job": "职务",
        "employees.sex": "性别",
        "employees.type": "职位类别",

        # 分支机构列表
        "branchs.name": "分支机构名称",
        "branchs.brRegNo": "分支机构企业注册号",
        "branchs.brPrincipal": "分支机构负责人",
        "branchs.cbuItem": "一般经营项目",
        "branchs.brAddr": "分支机构地址",

        # 变更记录列表
        "changes.type": "变更内容",
        "changes.beforeContent": "变更前内容",
        "changes.afterContent": "变更后内容",
        "changes.changeDate": "变更日期",

        # 经营异常信息
        "changemess.inreason": "列入经营异常名录原因",
        "changemess.indate": "列入日期",
        "changemess.outreason": "移出经营异常名录原因",
        "changemess.outdate": "移出日期",
        "changemess.belongorg": "作出列入决定机关",
        "changemess.outOrgan": "作出移出决定机关",

        # 年报信息
        "reports.annualreport": "报送年度",
        "reports.releasedate": "发布日期",

        # 严重违法失信企业名单
        "illegals.order": "序号",
        "illegals.type": "类别",
        "illegals.reason": "列入原因",
        "illegals.date": "列入日期",
        "illegals.organ": "列入决定机关",
        "illegals.reasonOut": "移出原因",
        "illegals.dateOut": "移出日期",
        "illegals.organOut": "移出决定机关",
    }
    return map_fields(data.get("result", {}), field_mapping)


async def annual_report_info(keyword, time_out=100):
    '''查询企业年报信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/annual-report-info?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    field_mapping = {
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
    }
    return [map_fields(item, field_mapping) for item in data.get("data", [])]


async def company_black_list(keyword, time_out=100):
    '''查询工商严重违法信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/company-black-list?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    mapping = {
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
    }
    return map_fields(data.get('data', {}), mapping)


async def saic_basic_info(keyword, time_out=100):
    '''查询工商基本信息，包括股东、成员、分支机构、变更记录等'''
    url = f'https://{Ideatech_Host}/cloudidp/api/saic-basic-info?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get("data", data.get("result", data))


async def final_beneficiary(keyword, time_out=100):
    '''查询目标企业最终受益人的情况'''
    url = f'https://{Ideatech_Host}/cloudidp/outer/finalBeneficiary?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get("data", data.get("result", data))


async def equity_share_list(keyword, time_out=100):
    '''查询企业股权结构（列表）情况'''
    url = f'https://{Ideatech_Host}/cloudidp/outer/equityShareList?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get("data", data.get("result", data))


async def base_account_record(keyword, time_out=100):
    '''查询基本户的履历状态信息，包括现在和历史记录'''
    url = f'https://{Ideatech_Host}/cloudidp/api/base-account-record?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get("data", data.get("result", data))


async def company_stock_relation(keyword, name='王思聪', time_out=100):
    '''公司名 + 主要人员姓名，查询对外投资/任职/法人情况'''
    url = f'https://{Ideatech_Host}/cloudidp/api/company-stock-relation?key={Config.Ideatech_API_Key}&keyWord={keyword}&name={name}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get("data", data.get("result", data))


async def company_out_investment(keyword, time_out=100):
    '''查询企业对外投资情况'''
    url = f'https://{Ideatech_Host}/cloudidp/api/company-out-investment?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get("data", data.get("result", data))


# async def micro_enterprise_identity(keyword, time_out=100):
#     '''识别企业是否为小微企业'''
#     url = f'https://{Ideatech_Host}/cloudidp/api/identity/microEnt?key={Config.Ideatech_API_Key}&keyWord={keyword}'
#     data = await call_http_request(url, headers=None, time_out=time_out)
#     return data.get("data", data.get("result", data))


async def simple_cancellation(keyword, time_out=100):
    '''企业简易注销公告查询'''
    url = f'https://{Ideatech_Host}/cloudidp/api/simple-cancellation?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get("data", data.get("result", data))


async def company_exception_list(keyword, time_out=100):
    '''查询工商经营异常名录信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/company-exception-list?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get("data", data.get("result", data))


async def company_personnel_risk(co_name='北京百度网讯科技有限公司', personnelname='梁志祥', time_out=100):
    '''通过企业名称和人员姓名查询人员相关风险列表'''
    url = f'https://{Ideatech_Host}/cloudidp/api/company-personnel-risk?key={Config.Ideatech_API_Key}&companyName={co_name}&companyPersonnelName={personnelname}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    return data.get('data', {})


# async def tax_arrears_info(keyword, time_out=100):
#     '''通过公司名称获取企业欠税信息，企业欠税信息包括欠税公告、纳税人识别号、证件号码、经营地点、欠税税种、欠税余额等字段的详细信息'''
#     url = f'https://{Ideatech_Host}/cloudidp/api/tax-arrears-info?key={Config.Ideatech_API_Key}&keyWord={keyword}'
#     data = await call_http_request(url, headers=None, time_out=time_out)
#     tax_arrears_mapping = {
#         "overduePeriod": "欠税所属期",
#         "pubDepartment": "发布单位",
#         "taxpayerType": "纳税人类型",
#         "pubDate": "发布日期",
#         "area": "所属市县区",
#         "address": "经营地点",
#         "operName": "负责人姓名",
#         "taxpayerNum": "纳税人识别号",
#         "overdueAmount": "欠税余额",
#         "overdueType": "欠税税种",
#         "operIdNum": "企业证照号",
#         "currOverdueAmount": "当前发生的欠税余额",
#     }
#     return [map_fields(item, tax_arrears_mapping) for item in data.get('data', [])]


async def court_notice_info(keyword, time_out=100):
    '''查询开庭公告信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/court-notice?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    litigation_mapping = {
        "defendant": "被告/被上诉人",
        "executegov": "法院",
        "prosecutor": "原告/上诉人",
        "courtDate": "开庭日期",
        "caseReason": "案由",
        "caseNo": "案号",
    }
    return [map_fields(item, litigation_mapping) for item in data.get('data', {}).get("result", [])]


async def judgment_doc(keyword, time_out=100):
    '''查询裁判文书'''
    url = f'https://{Ideatech_Host}/cloudidp/api/judgment-doc?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    judgement_mapping = {
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
    }
    return [map_fields(item, judgement_mapping) for item in data.get('data', {}).get("data", [])]


async def court_announcement(keyword, time_out=100):
    '''查询法院公告信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/court-announcement?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    court_notice_mapping = {
        "court": "执行法院",
        "companyName": "公司名称",
        "sortTime": "发布时间",
        "body": "内容",
        "relatedParty": "相关当事人",
        "ggType": "公告类型",
    }
    return [map_fields(item, court_notice_mapping) for item in data.get('data', {}).get("data", [])]


async def dishonesty_info(keyword, time_out=100):
    '''查询失信信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/dishonesty?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    enforced_mapping = {
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
    }

    return [map_fields(item, enforced_mapping) for item in data.get('data', {}).get("data", [])]


async def implements_info(keyword, time_out=100):
    '''查询失信被执行信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/implements?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    execution_case_mapping = {
        "companyName": "公司名称",
        "pname": "被执行姓名",
        "sortTime": "立案时间",
        "caseNo": "案号",
        "court": "执行法院名称",
        "execMoney": "执行标的",
        "idcardNo": "身份证/组织机构代码",
    }

    return [map_fields(item, execution_case_mapping) for item in data.get('data', {}).get("data", [])]


async def stock_freeze(keyword, time_out=100):
    '''查询股权冻结信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/stock-freeze?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    equity_freeze_mapping = {
        "executedBy": "被执行人",
        "equityAmount": "股权数额",
        "enforcementCourt": "执行法院",
        "executionNoticeNum": "执行通知书文号",
        "status": "状态",
        "equityFreezeDetail.companyName": "相关企业名称",
        "equityFreezeDetail.executionMatters": "执行事项",
        "equityFreezeDetail.executionDocNum": "执行文书文号",
        "equityFreezeDetail.executionVerdictNum": "执行裁定书文号",
        "equityFreezeDetail.freezeStartDate": "冻结开始日期",
        "equityFreezeDetail.freezeEndDate": "冻结结束日期",
        "equityFreezeDetail.publicDate": "公示日期",
        "equityUnFreezeDetail.executionMatters": "执行事项",
        "equityUnFreezeDetail.unFreezeDate": "解除冻结日期",
        "equityUnFreezeDetail.publicDate": "公示日期",
        "equityUnFreezeDetail.thawOrgan": "解冻机关",
    }

    return [map_fields(item, equity_freeze_mapping) for item in data.get("data", [])]


async def case_filing(keyword, time_out=100):
    '''查询立案信息'''
    url = f'https://{Ideatech_Host}/cloudidp/api/case-filing?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    court_case_mapping = {
        "pageSize": "每页条数",
        "pageIndex": "第几页",
        "totalRecords": "总记录数",
        "result": "案件详情",
        "result.caseNo": "案号",
        "result.publishDate": "立案日期",
        "result.courtYear": "案件年份",
        "result.prosecutorList": "原告列表",
        "result.prosecutorList.name": "原告名称",
        "result.defendantList": "被告列表",
        "result.defendantList.name": "被告名称",
    }

    return [map_fields(item, court_case_mapping) for item in data.get("data", {}).get("result", [])]


async def shell_company(keyword, time_out=100):
    '''通过分析企业的基本信息、日常经营信息等识别空壳企业'''
    url = f'https://{Ideatech_Host}/cloudidp/api/shellCompany?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await call_http_request(url, headers=None, time_out=time_out)
    shell_company_mapping = {
        "hasShellTags": "是否空壳",
        "shellTagsCount": "空壳特征数量",
        "tagsDetail": "空壳特征明细",
        "tagsDetail.tagAlias": "特征简称",
        "tagsDetail.tagTitle": "特征标题",
        "tagsDetail.description": "情况描述",
    }

    return map_fields(data.get("data", {}), shell_company_mapping)


if __name__ == "__main__":
    from utils import get_module_functions

    funcs = get_module_functions('agents.ai_company')
    print([i[0] for i in funcs])

    __all__ = ['annual_report_info', 'base_account_record', 'case_filing', 'company_black_list',
               'company_exception_list', 'company_out_investment', 'company_personnel_risk', 'company_stock_relation',
               'court_announcement', 'court_notice_info', 'dishonesty_info', 'equity_share_list', 'exact_saic_info',
               'final_beneficiary',
               'implements_info', 'judgment_doc', 'real_time_saic_info', 'saic_basic_info',
               'shell_company', 'simple_cancellation', 'stock_freeze']

    import nest_asyncio

    nest_asyncio.apply()


    # Config.load('../config.yaml')

    async def test():
        res = await exact_saic_info(keyword='小米科技有限责任公司', time_out=100)
        print(res)


    asyncio.run(test())
