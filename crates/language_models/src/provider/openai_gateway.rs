use anyhow::Result;
use collections::BTreeMap;
use credentials_provider::CredentialsProvider;
use futures::{AsyncReadExt, FutureExt, StreamExt, future::BoxFuture};
use gpui::{App, AppContext, AsyncApp, Context, Entity, SharedString, Task};
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest, RequestBuilderExt, http,
};
use language_model::chat_completion::{ChatCompletionEventMapper, ResponseStreamEvent};
use language_model::{
    ApiKeyConfiguration, ApiKeyState, AuthenticateError, EnvVar, IconOrSvg, LanguageModel,
    LanguageModelClient, LanguageModelCompletionError, LanguageModelCompletionStream,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoiceSupport, ModelRateLimiters, ProviderSettingsView, RateLimiter,
    unavailable_error,
};
use serde::Deserialize;
pub use settings::OpenAiCompatibleModelCapabilities as ModelCapabilities;
pub use settings::OpenAiGatewayAvailableModel as AvailableModel;
use settings::SettingsStore;
use std::marker::PhantomData;
use std::sync::Arc;
use ui::IconName;

/// Runtime settings shared by OpenAI-compatible gateway providers.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct OpenAiGatewaySettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub custom_headers: CustomHeaders,
}

/// Provider-specific constants and settings accessors for an OpenAI-compatible gateway.
pub trait OpenAiGatewayKind: Send + Sync + 'static {
    const PROVIDER_ID: LanguageModelProviderId;
    const PROVIDER_NAME: LanguageModelProviderName;
    const DEFAULT_API_URL: &'static str;
    const API_KEY_DOCS_URL: &'static str;
    const ICON: IconName;

    fn api_key_env_var() -> EnvVar;
    fn settings(cx: &App) -> &OpenAiGatewaySettings;
    fn default_available_model() -> AvailableModel;
}

pub struct OpenAiGatewayLanguageModelProvider<K: OpenAiGatewayKind> {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    request_limiters: ModelRateLimiters,
    _kind: PhantomData<K>,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    available_models: Vec<AvailableModel>,
    fetch_models_task: Option<Task<Result<(), LanguageModelCompletionError>>>,
    /// Cloned provider name for error reporting inside State methods.
    provider_name: LanguageModelProviderName,
    settings: fn(&App) -> &OpenAiGatewaySettings,
    default_api_url: &'static str,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key_state.has_key()
    }

    fn api_url(&self, cx: &App) -> SharedString {
        let api_url = &(self.settings)(cx).api_url;
        if api_url.is_empty() {
            self.default_api_url.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = self.api_url(cx);
        self.api_key_state.store(
            api_url,
            api_key,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        )
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = self.api_url(cx);
        let task = self.api_key_state.load_if_needed(
            api_url,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        );

        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| this.restart_fetch_models_task(cx))
                .ok();
            result
        })
    }

    fn fetch_models(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Task<Result<(), LanguageModelCompletionError>> {
        let http_client = self.http_client.clone();
        let api_url = self.api_url(cx);
        let api_key = self.api_key_state.key(&api_url);
        let extra_headers = (self.settings)(cx).custom_headers.clone();
        let provider_name = self.provider_name.clone();
        cx.spawn(async move |this, cx| {
            let models = list_models(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref(),
                &extra_headers,
                provider_name,
            )
            .await?;
            this.update(cx, |this, cx| {
                this.available_models = models;
                cx.notify();
            })
            .map_err(|e| LanguageModelCompletionError::Other(e))?;
            Ok(())
        })
    }

    fn restart_fetch_models_task(&mut self, cx: &mut Context<Self>) {
        if self.is_authenticated() {
            let task = self.fetch_models(cx);
            self.fetch_models_task.replace(task);
        } else {
            self.available_models = Vec::new();
        }
    }
}

impl<K: OpenAiGatewayKind> OpenAiGatewayLanguageModelProvider<K> {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>({
                let mut last_settings = K::settings(cx).clone();
                move |this: &mut State, cx| {
                    let current_settings = K::settings(cx);
                    if current_settings != &last_settings {
                        last_settings = current_settings.clone();
                        this.authenticate(cx).detach();
                        cx.notify();
                    }
                }
            })
            .detach();
            State {
                api_key_state: ApiKeyState::new(Self::api_url(cx), K::api_key_env_var()),
                credentials_provider,
                http_client: http_client.clone(),
                available_models: Vec::new(),
                fetch_models_task: None,
                provider_name: K::PROVIDER_NAME,
                settings: K::settings,
                default_api_url: K::DEFAULT_API_URL,
            }
        });

        Self {
            http_client,
            state,
            request_limiters: ModelRateLimiters::default(),
            _kind: PhantomData,
        }
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &K::settings(cx).api_url;
        if api_url.is_empty() {
            K::DEFAULT_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }

    /// Every model this provider offers, keyed by id: the default model, then
    /// the fetched list, then settings entries, each overriding the last.
    fn gateway_models(&self, cx: &App) -> BTreeMap<String, AvailableModel> {
        let mut models = BTreeMap::default();

        let default_model = K::default_available_model();
        models.insert(default_model.name.clone(), default_model);

        for model in self.state.read(cx).available_models.clone() {
            models.insert(model.name.clone(), model);
        }

        for model in &K::settings(cx).available_models {
            models.insert(model.name.clone(), model.clone());
        }

        models
    }

    /// The current configuration of `model`, if this provider still offers it.
    fn config(
        &self,
        model: &LanguageModel,
        cx: &App,
    ) -> Result<AvailableModel, LanguageModelCompletionError> {
        self.gateway_models(cx)
            .remove(model.id.0.as_ref())
            .ok_or_else(|| unavailable_error(model))
    }

    fn stream_open_ai(
        &self,
        request_limiter: &RateLimiter,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let (api_key, api_url, extra_headers) = self.state.read_with(cx, |state, cx| {
            let api_url = state.api_url(cx);
            let extra_headers = (state.settings)(cx).custom_headers.clone();
            (state.api_key_state.key(&api_url), api_url, extra_headers)
        });

        let future = request_limiter.stream(async move {
            let provider = K::PROVIDER_NAME;
            let Some(api_key) = api_key else {
                return Err(LanguageModelCompletionError::NoApiKey { provider });
            };
            let request = open_ai::stream_completion(
                http_client.as_ref(),
                provider.0.as_str(),
                &api_url,
                &api_key,
                request,
                &extra_headers,
            );
            let response = request
                .await
                .map_err(|error| map_open_ai_error(error, provider))?;
            Ok(response)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl<K: OpenAiGatewayKind> LanguageModelProviderState for OpenAiGatewayLanguageModelProvider<K> {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl<K: OpenAiGatewayKind> LanguageModelProvider for OpenAiGatewayLanguageModelProvider<K> {
    fn id(&self) -> LanguageModelProviderId {
        K::PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        K::PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(K::ICON)
    }

    fn default_model(&self, cx: &App) -> Option<LanguageModel> {
        self.gateway_models(cx)
            .get(&K::default_available_model().name)
            .map(|model| language_model::<K>(model))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<LanguageModel> {
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<LanguageModel> {
        self.gateway_models(cx)
            .values()
            .map(|model| language_model::<K>(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.read(cx);
        Some(ProviderSettingsView::ApiKey(ApiKeyConfiguration::new(
            state.api_key_state.has_key(),
            state.api_key_state.is_from_env_var(),
            state.api_key_state.env_var_name().clone(),
            K::API_KEY_DOCS_URL.into(),
        )))
    }

    fn set_api_key(&self, api_key: Option<String>, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.set_api_key(api_key, cx))
    }
}

impl<K: OpenAiGatewayKind> LanguageModelClient for OpenAiGatewayLanguageModelProvider<K> {
    fn stream_completion(
        &self,
        model: &LanguageModel,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<LanguageModelCompletionStream, LanguageModelCompletionError>>
    {
        let config = match cx.update(|cx| self.config(model, cx)) {
            Ok(config) => config,
            Err(error) => return async move { Err(error) }.boxed(),
        };
        let request_limiter = self.request_limiters.for_model(&model.id);
        let request = match crate::provider::open_ai::into_open_ai(
            request,
            &config.name,
            config.capabilities.parallel_tool_calls,
            config.capabilities.prompt_cache_key,
            config.max_output_tokens,
            crate::provider::open_ai::ChatCompletionMaxTokensParameter::MaxCompletionTokens,
            None,
            false,
        ) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };
        let completions = self.stream_open_ai(&request_limiter, request, cx);
        let executor = cx.background_executor().clone();
        async move {
            let mapper = ChatCompletionEventMapper::new();
            Ok(language_model::stream_in_background(
                mapper.map_stream(completions.await?).boxed(),
                executor,
            ))
        }
        .boxed()
    }
}

fn map_open_ai_error(
    error: open_ai::RequestError,
    provider: LanguageModelProviderName,
) -> LanguageModelCompletionError {
    match error {
        open_ai::RequestError::HttpResponseError {
            status_code,
            body,
            headers,
            ..
        } => {
            let retry_after = headers
                .get(http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()?.parse::<u64>().ok())
                .map(std::time::Duration::from_secs);

            LanguageModelCompletionError::from_http_status(
                provider,
                status_code,
                extract_error_message(&body),
                retry_after,
            )
        }
        error => error.into(),
    }
}

fn extract_error_message(body: &str) -> String {
    let json = match serde_json::from_str::<serde_json::Value>(body) {
        Ok(json) => json,
        Err(_) => return body.to_string(),
    };

    json.get("error")
        .and_then(|value| {
            value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .or_else(|| value.as_str())
        })
        .or_else(|| json.get("message").and_then(serde_json::Value::as_str))
        .map(ToString::to_string)
        .unwrap_or_else(|| body.to_string())
}

fn has_tag(tags: &[String], expected: &str) -> bool {
    tags.iter()
        .any(|tag| tag.trim().eq_ignore_ascii_case(expected))
}

fn language_model<K: OpenAiGatewayKind>(model: &AvailableModel) -> LanguageModel {
    LanguageModel {
        supports_tools: model.capabilities.tools,
        supports_images: model.capabilities.images,
        tool_choice_support: LanguageModelToolChoiceSupport {
            auto: model.capabilities.tools,
            any: model.capabilities.tools,
            none: true,
        },
        supports_streaming_tools: true,
        supports_split_token_display: true,
        max_output_tokens: model.max_output_tokens,
        ..LanguageModel::new(
            LanguageModelId::from(model.name.clone()),
            LanguageModelName::from(
                model
                    .display_name
                    .clone()
                    .unwrap_or_else(|| model.name.clone()),
            ),
            K::PROVIDER_ID,
            K::PROVIDER_NAME,
            format!("{}/{}", K::PROVIDER_ID.0, model.name),
            model.max_tokens,
        )
    }
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ApiModel>,
}

#[derive(Deserialize)]
struct ApiModel {
    id: String,
    name: Option<String>,
    context_window: Option<u64>,
    max_tokens: Option<u64>,
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    supported_parameters: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    architecture: Option<ApiModelArchitecture>,
}

#[derive(Deserialize)]
struct ApiModelArchitecture {
    #[serde(default)]
    input_modalities: Vec<String>,
}

async fn list_models(
    client: &dyn HttpClient,
    api_url: &str,
    api_key: Option<&str>,
    extra_headers: &CustomHeaders,
    provider_name: LanguageModelProviderName,
) -> Result<Vec<AvailableModel>, LanguageModelCompletionError> {
    let uri = format!("{api_url}/models");
    let mut request_builder = HttpRequest::builder()
        .method(Method::GET)
        .uri(uri)
        .header("Accept", "application/json");
    if let Some(api_key) = api_key {
        request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
    }
    let request = request_builder
        .extra_headers(extra_headers)
        .body(AsyncBody::default())
        .map_err(|error| LanguageModelCompletionError::BuildRequestBody {
            provider: provider_name.clone(),
            error,
        })?;
    let host = request.uri().host().unwrap_or(api_url).to_owned();
    let mut response =
        client
            .send(request)
            .await
            .map_err(|error| LanguageModelCompletionError::HttpSend {
                provider: provider_name.clone(),
                host,
                error,
            })?;

    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .map_err(|error| LanguageModelCompletionError::ApiReadResponseError {
            provider: provider_name.clone(),
            error,
        })?;

    if !response.status().is_success() {
        return Err(LanguageModelCompletionError::from_http_status(
            provider_name.clone(),
            response.status(),
            extract_error_message(&body),
            None,
        ));
    }

    let response: ModelsResponse = serde_json::from_str(&body).map_err(|error| {
        LanguageModelCompletionError::DeserializeResponse {
            provider: provider_name,
            error,
        }
    })?;

    let mut models = Vec::new();
    for model in response.data {
        if let Some(model_type) = model.r#type.as_deref()
            && model_type != "language"
        {
            continue;
        }

        let has_capability_metadata =
            !model.supported_parameters.is_empty() || !model.tags.is_empty();
        let supports_tools = if has_capability_metadata {
            model
                .supported_parameters
                .iter()
                .any(|parameter| parameter == "tools")
                || has_tag(&model.tags, "tool-use")
                || has_tag(&model.tags, "tools")
        } else {
            // NVIDIA NIM / OmniRoute often omit capability tags; assume tools work.
            true
        };
        let supports_images = model.architecture.is_some_and(|architecture| {
            architecture
                .input_modalities
                .iter()
                .any(|modality| modality == "image")
        }) || has_tag(&model.tags, "vision")
            || has_tag(&model.tags, "image-input");
        let parallel_tool_calls = model
            .supported_parameters
            .iter()
            .any(|parameter| parameter == "parallel_tool_calls");
        let prompt_cache_key = model
            .supported_parameters
            .iter()
            .any(|parameter| parameter == "prompt_cache_key" || parameter == "cache_control");
        models.push(AvailableModel {
            name: model.id.clone(),
            display_name: model.name.or(Some(model.id)),
            max_tokens: model.context_window.or(model.max_tokens).unwrap_or(128_000),
            max_output_tokens: model.max_tokens,
            max_completion_tokens: None,
            capabilities: ModelCapabilities {
                tools: supports_tools,
                images: supports_images,
                parallel_tool_calls,
                prompt_cache_key,
                chat_completions: true,
                interleaved_reasoning: false,
                max_tokens_parameter: false,
            },
        });
    }

    Ok(models)
}
