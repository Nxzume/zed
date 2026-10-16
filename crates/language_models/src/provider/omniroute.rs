use gpui::App;
use language_model::{EnvVar, LanguageModelProviderId, LanguageModelProviderName, env_var};
use settings::Settings;
use std::sync::LazyLock;
use ui::IconName;

use crate::provider::openai_gateway::{
    AvailableModel, ModelCapabilities, OpenAiGatewayKind, OpenAiGatewayLanguageModelProvider,
    OpenAiGatewaySettings,
};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("omniroute");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("OmniRoute");

const API_KEY_ENV_VAR_NAME: &str = "OMNIROUTE_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

pub struct OmniRoute;

impl OpenAiGatewayKind for OmniRoute {
    const PROVIDER_ID: LanguageModelProviderId = PROVIDER_ID;
    const PROVIDER_NAME: LanguageModelProviderName = PROVIDER_NAME;
    const DEFAULT_API_URL: &'static str = "http://localhost:20128/v1";
    const API_KEY_DOCS_URL: &'static str = "https://github.com/diegosouzapw/OmniRoute";
    const ICON: IconName = IconName::AiOpenAiCompat;

    fn api_key_env_var() -> EnvVar {
        (*API_KEY_ENV_VAR).clone()
    }

    fn settings(cx: &App) -> &OpenAiGatewaySettings {
        &crate::AllLanguageModelSettings::get_global(cx).omniroute
    }

    fn default_available_model() -> AvailableModel {
        AvailableModel {
            name: "auto".to_string(),
            display_name: Some("Auto".to_string()),
            max_tokens: 128_000,
            max_output_tokens: None,
            max_completion_tokens: None,
            capabilities: ModelCapabilities {
                tools: true,
                images: false,
                parallel_tool_calls: false,
                prompt_cache_key: false,
                chat_completions: true,
                interleaved_reasoning: false,
                max_tokens_parameter: false,
            },
        }
    }
}

pub type OmniRouteLanguageModelProvider = OpenAiGatewayLanguageModelProvider<OmniRoute>;

pub type OmniRouteSettings = OpenAiGatewaySettings;
