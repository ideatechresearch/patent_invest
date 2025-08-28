import requests, httpx, json
from utils import map_fields
from service import get_httpx_client, call_http_request, async_error_logger
from config import Config


@async_error_logger(max_retries=2, delay=3, exceptions=(Exception, httpx.HTTPError))
async def fetch_url_retry(url, timeout=30):
    cx = get_httpx_client(time_out=Config.HTTP_TIMEOUT_SEC)
    result = await call_http_request(url=url, headers=None, timeout=timeout, httpx_client=cx)

    if result.get('status') == "fail" and result.get("code") == "000205":  # "数据正在计算，请稍后再试
        raise Exception(f"[{url}] 接口失败，原因：{result.get('reason', result.get('text'))}")  # 主动抛出业务异常，让装饰器重试

    return result


async def exact_saic_info(keyword, time_out=100):
    '''查询工商全维度信息（年检比对等批量调用场景使用）'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/outer/exactSaicInfo?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
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
    url = f'https://{Config.Ideatech_Host}/cloudidp/outer/realTimeSaicInfo?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
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
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/annual-report-info?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
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
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/company-black-list?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
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
    return map_fields(data.get('data', {}), mapping) or data


async def saic_basic_info(keyword, time_out=100):
    '''查询工商基本信息，包括股东、成员、分支机构、变更记录等'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/saic-basic-info?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
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
        "employees.name": "员工姓名",
        "employees.job": "员工职务",

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
    }

    return map_fields(data.get('data', {}), field_mapping) or data


async def saic_basic_legal_person(keyword, time_out=100):
    '''查询工商基本信息查询法人股东'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/saic-basic-info?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    raw_data = data.get('data', {})
    legal = raw_data.get('legalPerson') or raw_data.get('legalPersonName')
    return legal.strip() if legal else None


async def final_beneficiary(keyword, time_out=100):
    '''查询目标企业最终受益人的情况'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/outer/finalBeneficiary?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
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
    }
    return map_fields(data.get('result', {}), field_mapping) or data


async def equity_share_list(keyword, time_out=100):
    '''查询企业股权结构（列表）情况'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/outer/equityShareList?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
        # 顶层字段
        "total": "总记录条数",

        # 嵌套列表字段：result.item
        "item.name": "企业名称或者人名",
        "item.type": "股东类型",
        "item.layer": "层级",
        "item.percent": "股份占比",
        "item.capital": "出资额",
        "item.parent": "父亲节点"
    }
    return map_fields(data.get('result', {}), field_mapping) or data


async def base_account_record(keyword, time_out=100):
    '''查询基本户的履历状态信息，包括现在和历史记录'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/base-account-record?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
        "total": "总记录条数",

        # items 是列表中的字段，采用 items.xxx 的格式表示
        "items.name": "单位名称/账户名称",
        "items.licenseKey": "基本户许可证号",
        "items.licenseOrg": "审批机关",
        "items.licenseDate": "审批日期",
        "items.licenseType": "许可类型"
    }
    return map_fields(data.get('data', {}), field_mapping)


async def company_stock_relation(keyword, name='王思聪', time_out=100):
    '''公司名 + 主要人员姓名，查询对外投资/任职/法人情况'''
    if not name:
        name = await saic_basic_legal_person(keyword, time_out=time_out)
    if not name:
        return {'error': '无法从工商基本信息中提取法人信息'}
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/company-stock-relation?key={Config.Ideatech_API_Key}&keyWord={keyword}&name={name}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
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
    }

    return map_fields(data.get('data', {}), field_mapping)


async def company_stock_deep_relation(keyword, limit: int = 5, time_out=100):
    """
    企业法人对外投资信息深度分析数据源获取实现：
    1. 获取法人信息；
    2. 查询法人担任法人公司的信息（companyLegal）；
    3. 提取前 N 个法人公司名称；
    4. 查询每个企业的工商基本信息；
    """
    legal_person = await saic_basic_legal_person(keyword, time_out=time_out)
    if not legal_person:
        return {'error': '无法从工商基本信息中提取法人信息'}
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/company-stock-relation?key={Config.Ideatech_API_Key}&keyWord={keyword}&name={legal_person}'
    data = await fetch_url_retry(url, timeout=time_out)
    company_legal = [item.get('name') for item in data.get('data', {}).get('companyLegal', []) if item.get('name')]
    company_names = list(dict.fromkeys(company_legal))[:limit]  # 去重 + 截取前N个
    if not company_names:
        return {'error': '未找到法人企业列表'}
    return [await saic_basic_info(company, time_out) for company in company_names]


async def company_out_investment(keyword, time_out=100):
    '''查询企业对外投资情况'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/company-out-investment?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
        "name": "公司名称",
        "no": "注册号",
        "creditCode": "全国统一信用代码",
        "econKind": "企业类型",
        "status": "状态",
        "registCapi": "注册资本",
        "operName": "法人",
        "fundedRatio": "出资比例",
        "startDate": "成立日期"
    }
    return [map_fields(item, field_mapping) for item in data.get('data', {}).get("companyOutInvestment", [])]


# async def micro_enterprise_identity(keyword, time_out=100):
#     '''识别企业是否为小微企业'''
#     url = f'https://{Config.Ideatech_Host}/cloudidp/api/identity/microEnt?key={Config.Ideatech_API_Key}&keyWord={keyword}'
#     data = await fetch_url_retry(url, timeout=time_out)
#     return data.get("data", data.get("result", data))


async def simple_cancellation(keyword, time_out=100):
    '''企业简易注销公告查询'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/simple-cancellation?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
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
    }

    return map_fields(data.get('data', {}), field_mapping) or data


async def company_exception_list(keyword, time_out=100):
    '''查询工商经营异常名录信息'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/company-exception-list?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
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
    }

    return map_fields(data.get('data', {}), field_mapping) or data


async def company_personnel_risk(co_name='北京百度网讯科技有限公司', name='梁志祥', time_out=100):
    '''通过企业名称和人员姓名查询人员相关风险列表'''
    if not name:
        name = await saic_basic_legal_person(co_name, time_out=time_out)
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/company-personnel-risk?key={Config.Ideatech_API_Key}&companyName={co_name}&companyPersonnelName={name}'
    data = await fetch_url_retry(url, timeout=time_out)
    field_mapping = {
        "count": "风险总数",
        "riskClassification": "风险分类",  # 周边风险，预警提醒，自身风险，历史风险

        "list.total": "风险项总数",
        "list.tag": "风险标签",  # 警示,高风险
        "list.title": "风险标题",  # 立案信息
        "list.type": "风险类型",

        "list.list.companyName": "企业名称",
        "list.list.title": "风险明细标题",
        "list.list.type": "风险明细类型",
        "list.list.riskCount": "风险明细数量",
        "list.list.desc": "风险描述"  # 股权出质,简易注销,清算信息,立案信息_人员
    }
    type_mapping = {
        1: "严重违法",
        3: "失信被执行人（公司）",
        5: "被执行人（公司）",
        6: "行政处罚",
        7: "经营异常",
        8: "法律诉讼",
        9: "股权出质（公司）",
        10: "动产抵押",
        11: "欠税公告",
        12: "名称变更",
        13: "开庭公告",
        14: "法院公告",
        15: "法人变更",
        16: "投资人变更",
        17: "主要人员变更",
        18: "注册资本变更",
        19: "注册地址变更",
        20: "出资情况变更",
        21: "司法协助（公司）",
        22: "清算信息",
        23: "知识产权出质",
        24: "环保处罚",
        25: "公示催告",
        26: "送达公告",
        27: "立案信息",
        28: "税收违法",
        29: "司法拍卖",
        30: "土地抵押",
        31: "简易注销",
        32: "限制消费令（公司）",
        33: "限制消费令（人）",
        34: "终本案件",
        35: "股权出质（人）",
        36: "司法协助（人）",
        37: "股权质押（人）",
        38: "破产案件",
        39: "询价评估",
        40: "抽查检查",
        41: "对外担保",
        42: "违规处理",
        45: "强制清算",
        46: "终本案件（人）",
        47: "开庭公告（人）",
        48: "法院公告（人）",
        49: "送达公告（人）",
        50: "立案信息（人）",
        51: "股权质押",
        53: "严重违法（已移出）",
        55: "经营异常（已移出）",
        56: "法律诉讼（人）",
        62: "涉金融黑名单",
        63: "注销备案",
        64: "食品安全",
        65: "产品召回",
        80: "历史终本案件（人）",
        81: "历史司法协助（人）",
        82: "历史股权出质（人）"
    }

    return [map_fields(item, field_mapping) for item in data.get('data', [])] or data


# async def tax_arrears_info(keyword, time_out=100):
#     '''通过公司名称获取企业欠税信息，企业欠税信息包括欠税公告、纳税人识别号、证件号码、经营地点、欠税税种、欠税余额等字段的详细信息'''
#     url = f'https://{Config.Ideatech_Host}/cloudidp/api/tax-arrears-info?key={Config.Ideatech_API_Key}&keyWord={keyword}'
#     data = await fetch_url_retry(url, timeout=time_out)
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
#
# async def tax_case(keyword, time_out=100):
#     '''通过公司名称获取税收违法信息，税收违法信息包括纳税人名称、案件性质等字段的详细信息'''
#     url = f'https://{Config.Ideatech_Host}/cloudidp/api/tax-case?key={Config.Ideatech_API_Key}&keyWord={keyword}'
#     data = await fetch_url_retry(url, timeout=time_out)
#     field_mapping = {
#         "property": "案件性质",  # 税收异常非正常户
#         "name": "纳税人名称",
#         "time": "发生时间"
#     }
#
#     return [map_fields(item, field_mapping) for item in data.get('data', [])] or data


async def court_notice_info(keyword, time_out=100):
    '''查询开庭公告信息'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/court-notice?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    litigation_mapping = {
        "defendant": "被告/被上诉人",
        "executegov": "法院",
        "prosecutor": "原告/上诉人",
        "courtDate": "开庭日期",
        "caseReason": "案由",
        "caseNo": "案号",
    }
    return [map_fields(item, litigation_mapping) for item in data.get('data', {}).get("result", [])] or data


async def judgment_doc(keyword, time_out=100):
    '''查询裁判文书'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/judgment-doc?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
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
    return [map_fields(item, judgement_mapping) for item in data.get('data', {}).get("data", [])] or data


async def court_announcement(keyword, time_out=100):
    '''查询法院公告信息'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/court-announcement?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    court_notice_mapping = {
        "court": "执行法院",
        "companyName": "公司名称",
        "sortTime": "发布时间",
        "body": "内容",
        "relatedParty": "相关当事人",
        "ggType": "公告类型",  # 裁判文书,起诉状副本及开庭传票
    }
    return [map_fields(item, court_notice_mapping) for item in data.get('data', {}).get("data", [])] or data


async def dishonesty_info(keyword, time_out=100):
    '''查询失信信息'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/dishonesty?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
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

    return [map_fields(item, enforced_mapping) for item in data.get('data', {}).get("data", [])] or data


async def implements_info(keyword, time_out=100):
    '''查询失信被执行信息'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/implements?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    execution_case_mapping = {
        "companyName": "公司名称",
        "pname": "被执行姓名",
        "sortTime": "立案时间",
        "caseNo": "案号",
        "court": "执行法院名称",
        "execMoney": "执行标的",
        "idcardNo": "身份证/组织机构代码",
    }

    return [map_fields(item, execution_case_mapping) for item in data.get('data', {}).get("data", [])] or data


async def stock_freeze(keyword, time_out=100):
    '''查询股权冻结信息'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/stock-freeze?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)

    # # 股权冻结情况
    # equity_freeze_mapping = {
    #     "companyName": "相关企业名称",
    #     "executionMatters": "执行事项",
    #     "executionDocNum": "执行文书文号",
    #     "executionVerdictNum": "执行裁定书文号",
    #     "executedPersonDocType": "被执行人证件种类",
    #     "executedPersonDocNum": "被执行人证件号码",
    #     "freezeStartDate": "冻结开始日期",
    #     "freezeEndDate": "冻结结束日期",
    #     "freezeTerm": "冻结期限",
    #     "publicDate": "公示日期"
    # }
    # # 解除冻结详情
    # equity_unfreeze_mapping = {
    #     "executionMatters": "执行事项",
    #     "executionVerdictNum": "执行裁定书文号",
    #     "executionDocNum": "执行文书文号",
    #     "executedPersonDocType": "被执行人证件种类",
    #     "executedPersonDocNum": "被执行人证件号码",
    #     "unFreezeDate": "解除冻结日期",
    #     "publicDate": "公示日期",
    #     "thawOrgan": "解冻机关",
    #     "thawDocNo": "解冻文书号"
    # }
    # # 股东变更信息
    # judicial_change_mapping = {
    #     "executionMatters": "执行事项",
    #     "executionVerdictNum": "执行裁定书文号",
    #     "executedPersonDocType": "被执行人证件种类",
    #     "executedPersonDocNum": "被执行人证件号码",
    #     "assignee": "受让人",
    #     "assistExecDate": "协助执行日期",
    #     "assigneeDocKind": "受让人证件种类",
    #     "assigneeRegNo": "受让人证件号码",
    #     "stockCompanyName": "股权所在公司名称"
    # }
    field_mapping = {
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
    }
    return [map_fields(item, field_mapping) for item in data.get("data", [])] or data


async def case_filing(keyword, time_out=100):
    '''查询立案信息'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/case-filing?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    court_case_mapping = {
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
    }

    return [map_fields(item, court_case_mapping) for item in data.get("data", {})] or data


async def shell_company(keyword, time_out=100):
    '''通过分析企业的基本信息、日常经营信息等识别空壳企业'''
    url = f'https://{Config.Ideatech_Host}/cloudidp/api/shellCompany?key={Config.Ideatech_API_Key}&keyWord={keyword}'
    data = await fetch_url_retry(url, timeout=time_out)
    shell_company_mapping = {
        "hasShellTags": "是否空壳",
        "shellTagsCount": "空壳特征数量",
        "tagsDetail": "空壳特征明细",
        "tagsDetail.tagAlias": "特征简称",
        "tagsDetail.tagTitle": "特征标题",
        "tagsDetail.description": "情况描述",
    }

    return map_fields(data.get("data", {}), shell_company_mapping) or data


def get_cninfo_reports(stock_code, report_type='年报', start_date='', end_date=''):
    """
    从巨潮资讯网获取上市公司财报

    参数:
        stock_code: 股票代码(如'000001'表示平安银行)
        report_type: 报告类型('年报','季报','半年报'等)
        start_date: 开始日期(格式'YYYY-MM-DD')
        end_date: 结束日期(格式'YYYY-MM-DD')

    返回:
        DataFrame包含报告列表
    """
    import pandas as pd
    url = 'http://www.cninfo.com.cn/new/hisAnnouncement/query'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
    }

    # 构建查询参数
    params = {
        'stock': f'{stock_code}',
        'tabName': 'fulltext',
        'pageSize': '30',
        'pageNum': '1',
        'column': 'sse' if stock_code.startswith('6') else 'szse',
        'category': 'category_ndbg_szsh' if report_type == '年报' else 'category_bndbg_szsh',
        'seDate': f'{start_date}~{end_date}' if start_date and end_date else ''
    }

    try:
        response = requests.post(url, headers=headers, params=params)
        data = response.json()
        if data['announcements']:
            df = pd.DataFrame(data['announcements'])
            # 选择需要的列
            df = df[['announcementTitle', 'adjunctUrl', 'announcementTime']]
            # 添加下载链接前缀
            df['adjunctUrl'] = 'http://www.cninfo.com.cn' + df['adjunctUrl']
            return df
        else:
            print(f"未找到相关报告:{response.url}")
            return None
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def get_eastmoney_reports(stock_code, report_type='年报'):
    """
    从东方财富网获取上市公司财报
    #https://quantapi.eastmoney.com/Manual/Index?from=web&loc=%E6%8E%A5%E5%8F%A3%E9%85%8D%E7%BD%AE&ploc=%E6%8E%A5%E5%8F%A3%E9%85%8D%E7%BD%AE
    参数:
        stock_code: 股票代码(带市场前缀，如'SH600519')
        report_type: 报告类型('年报','季报'等)

    返回:
        DataFrame包含报告列表
    """
    import pandas as pd
    # 报告类型映射
    report_map = {
        '年报': 'category_ndbg_szsh;',
        '季报': 'category_yjdbg_szsh;',
        '半年报': 'category_bndbg_szsh;'
    }

    url = 'http://datacenter.eastmoney.com/securities/api/data/get'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
    }

    params = {
        'type': 'RPT_F10_FINANCE_FSATEMENT',
        'sty': 'ALL',
        'source': 'SECURITIES',
        'client': 'WEB',
        'filter': f'(SECURITY_CODE="{stock_code}")(REPORT_TYPE="{report_map.get(report_type, "")}")',
        'p': '1',
        'ps': '50'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if data['result'] and data['result']['data']:
            df = pd.DataFrame(data['result']['data'])
            # 选择需要的列
            keep_cols = ['SECURITY_NAME', 'REPORT_DATE', 'TITLE', 'URL']
            df = df[[col for col in keep_cols if col in df.columns]]
            return df
        else:
            print(f"未找到相关报告:{response.url}")
            return None
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


if __name__ == "__main__":
    from utils import get_module_functions

    # funcs = get_module_functions('agents.ai_company')
    # print([i[0] for i in funcs])

    __all__ = ['annual_report_info', 'base_account_record', 'case_filing', 'company_black_list',
               'company_exception_list', 'company_out_investment', 'company_personnel_risk', 'company_stock_relation',
               'court_announcement', 'court_notice_info', 'dishonesty_info', 'equity_share_list', 'exact_saic_info',
               'final_beneficiary',
               'implements_info', 'judgment_doc', 'real_time_saic_info', 'saic_basic_info',
               'company_stock_deep_relation',
               'shell_company', 'simple_cancellation', 'stock_freeze']

    import asyncio
    import nest_asyncio

    nest_asyncio.apply()

    Config.load('../config.yaml')


    async def test():
        res = await exact_saic_info(keyword='小米科技有限责任公司', time_out=100)
        print(res)


    asyncio.run(test())
    # 示例：获取贵州茅台(600519)2020-2022年的年报
    print(get_cninfo_reports('600519', '年报', '2022-01-01', '2024-12-31'))
    print(get_eastmoney_reports('SH600519', '年报'))
