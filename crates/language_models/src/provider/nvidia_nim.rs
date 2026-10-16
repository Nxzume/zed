use gpui::App;
use language_model::{EnvVar, LanguageModelProviderId, LanguageModelProviderName};
use settings::Settings;
use std::sync::LazyLock;
use ui::IconName;

use crate::provider::openai_gateway::{
    AvailableModel, ModelCapabilities, OpenAiGatewayKind, OpenAiGatewayLanguageModelProvider,
    OpenAiGatewaySettings,
};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("nvidia_nim");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("NVIDIA NIM");

static API_KEY_ENV_VAR: LazyLock<EnvVar> = LazyLock::new(|| {
    // Prefer NVIDIA_API_KEY (NVIDIA's documented name); fall back to provider-id form.
    EnvVar::new("NVIDIA_API_KEY".into()).or(EnvVar::new("NVIDIA_NIM_API_KEY".into()))
});

pub struct NvidiaNim;

impl OpenAiGatewayKind for NvidiaNim {
    const PROVIDER_ID: LanguageModelProviderId = PROVIDER_ID;
    const PROVIDER_NAME: LanguageModelProviderName = PROVIDER_NAME;
    const DEFAULT_API_URL: &'static str = "https://integrate.api.nvidia.com/v1";
    const API_KEY_DOCS_URL: &'static str = "https://build.nvidia.com/settings/api-keys";
    const ICON: IconName = IconName::AiOpenAiCompat;

    fn api_key_env_var() -> EnvVar {
        (*API_KEY_ENV_VAR).clone()
    }

    fn settings(cx: &App) -> &OpenAiGatewaySettings {
        &crate::AllLanguageModelSettings::get_global(cx).nvidia_nim
    }

    fn default_available_model() -> AvailableModel {
        AvailableModel {
            name: "meta/llama-3.1-70b-instruct".to_string(),
            display_name: Some("Llama 3.1 70B Instruct".to_string()),
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

pub type NvidiaNimLanguageModelProvider = OpenAiGatewayLanguageModelProvider<NvidiaNim>;

pub type NvidiaNimSettings = OpenAiGatewaySettings;
