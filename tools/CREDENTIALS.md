# 凭据配置（公开发布时已剔除，需自行填回）

本仓库在公开发布前已把 CANNJudge 的 JWT、开发机 SSH 密码 / hostkey 全部替换成占位符：

| 占位符 | 含义 | 怎么填 |
|---|---|---|
| `<CANNJUDGE_JWT_REDACTED>` | CANNJudge 登录 Cookie 的值 | 浏览器登录 cannjudge.cn → DevTools → Application → Cookies → 复制 `cannjudge_auth` 的值 |
| `<MY_USER_ID>` | 平台 userId | JWT 的 payload 里有 `uid` 字段（base64 解一下），或从任意 API 返回的 `user_id` 取 |
| `<DEV_HOST>` | 开发机地址 | 你自己的 910B 机器 IP |
| `<SSH_PASSWORD_REDACTED>` | 开发机 SSH 密码 | 你自己的密码（建议改用 key 认证） |
| `<SSH_HOSTKEY_REDACTED>` | 开发机 host key 指纹 | `ssh-keyscan -p <port> <host>` 或首次连接时确认 |

## 快捷用法

```bash
# 用环境变量覆盖，避免把凭据写进文件
export CJ_JWT="<你的 cannjudge_auth cookie 值>"
export CJ_USER="<你的 userId>"
```

然后在脚本里改成读环境变量即可，例如：

```python
import os
JWT = os.environ.get("CJ_JWT", "<CANNJUDGE_JWT_REDACTED>")
USER = os.environ.get("CJ_USER", "<MY_USER_ID>")
```

## 注意

- `cannjudge_auth` 是**账号级凭据**，泄露等于把提交配额交给别人（每日 50 次）。**不要提交到公开仓库**。
- 平台 API 一览（无需凭据也能读的部分）：`GET /api/problems/<pid>/ranking` —— 用于看别人成绩、反推判题口径。
